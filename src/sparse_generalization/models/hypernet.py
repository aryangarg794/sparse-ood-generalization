import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import zuko

from copy import deepcopy
from functools import partial
from torch import Tensor
from torch.nn.functional import softmax, gumbel_softmax
from torchmetrics.classification import BinaryAccuracy
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List

from sparse_generalization.layers.priors import LaplacePrior, NormalPrior
from sparse_generalization.losses.sparse_loss import L1SparsityAdjacency
from sparse_generalization.layers.vae import FlowVAE
from sparse_generalization.layers.priors import make_unit_gaussian
from sparse_generalization.utils.util_funcs import (
    positionalencoding2d,
    compute_attn_mean,
    compute_attn_mean_ens, 
    compute_mask_mean,
    compute_max_paths,
)

class HyperNet(nn.Module):

    def __init__(
        self,
        prior_func = make_unit_gaussian,
        prior_type: str = "uniform",
        num_mha_layers: int = 1,
        include_agg_layer: bool = False, 
        seq_len: int = 1,
        embed_size: int = 32, 
        out_dim: int = 1, 
        num_heads: int = 1,
        dropout: float = 0.0,
        hyper_type: str = "qk",
        flow_params: dict = {"n_flows": 3, "hidden_features": [256, 256]},
        prior_params: dict = {"n_flows": 3, "hidden_features": (256, 256)},
        residual: bool = False,
        device: str = "cuda",
        layernorm: bool = True,
        separate_mask: bool = False,
        use_mask: bool = False,
        act: nn.Module = nn.ReLU,
        force_vae_gaussian: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if embed_size % num_heads != 0:
            raise SyntaxError(
                f"Embed Size not divisible by number of heads, embed_size % num_heads = {embed_size % num_heads}"
            )
        self.hyper_type = hyper_type
        self.num_heads = num_heads
        self.residual = residual
        self.prior_type = prior_type
        self.seq_len = seq_len
        self.num_mha_layers = num_mha_layers
        self.include_agg_layer = include_agg_layer
        self.base_dist_size = 0
        self.agg_dist_size = 0
        self.device = device
        self.embed_size = embed_size
        self.layernorm = layernorm
        self.dk = embed_size // num_heads
        self.num_agg_layers = 1 if include_agg_layer else 0 

        if hyper_type == "mask":
            self.query_layers = nn.ModuleList([nn.Linear(embed_size, embed_size) for _ in range(num_mha_layers + self.num_agg_layers)])
            self.key_layers = nn.ModuleList([nn.Linear(embed_size, embed_size) for _ in range(num_mha_layers + self.num_agg_layers)])
            self.value_layers = nn.ModuleList([nn.Linear(embed_size, embed_size) for _ in range(num_mha_layers + self.num_agg_layers)])
            self.proj_layers = nn.ModuleList([nn.Linear(embed_size, embed_size) for _ in range(num_mha_layers + self.num_agg_layers)])

            self.base_dist_size = seq_len ** 2
            self.agg_dist_size = seq_len
            use_encoder = True
            encoder_heads = True

        elif hyper_type == "mha":
            self.base_dist_size = 4 * embed_size ** 2
            self.agg_dist_size = self.base_dist_size
            use_encoder = False
            encoder_heads = False

        elif hyper_type == "directa":
            self.value_layers = nn.ModuleList([nn.Linear(embed_size, embed_size) for _ in range(num_mha_layers + self.num_agg_layers)])
            self.proj_layers = nn.ModuleList([nn.Linear(embed_size, embed_size) for _ in range(num_mha_layers + self.num_agg_layers)])

            self.base_dist_size = seq_len ** 2
            self.agg_dist_size = seq_len
            use_encoder = True
            encoder_heads = True

        elif hyper_type == "qk":
            self.value_layers = nn.ModuleList([nn.Linear(embed_size, embed_size) for _ in range(num_mha_layers + self.num_agg_layers)])
            self.proj_layers = nn.ModuleList([nn.Linear(embed_size, embed_size) for _ in range(num_mha_layers + self.num_agg_layers)])

            self.base_dist_size = 2 * embed_size ** 2
            self.agg_dist_size = self.base_dist_size
            use_encoder = False
            encoder_heads = False

        self.queries = nn.init.uniform_(nn.Parameter(torch.zeros((1, embed_size), device=device)))
        self.total_mha_size = self.num_mha_layers * self.base_dist_size
        self.total_agg_size = self.num_agg_layers * self.agg_dist_size 
        self.total_num_layers = self.num_agg_layers + self.num_mha_layers
        self.total_dist_size = self.total_mha_size + self.total_agg_size

        if self.prior_type == "nf":
            self.prior = zuko.flows.MAF(
                features=self.total_dist_size,
                transforms=prior_params["n_flows"],
                hidden_features=prior_params["hidden_features"],
            )
        elif self.prior_type == "laplace":
            self.prior = LaplacePrior()
        elif self.prior_type == "normal":
            self.prior = NormalPrior()
        else:
            self.prior = nn.Identity()

        self.param_flow = FlowVAE(
            input_dim=embed_size, 
            base_dist=prior_func(self.total_dist_size),
            output_dim=self.total_dist_size,
            num_heads=num_heads, 
            encoder_heads=encoder_heads, 
            use_encoder=use_encoder,
            use_mask=use_mask, 
            separate_mask=separate_mask,
            flow_params=flow_params,
            layernorm=layernorm,
            force_vae_gaussian=force_vae_gaussian,
            device=device
        )

        self.ln1s = nn.ModuleList([nn.LayerNorm(embed_size) for _ in range(num_mha_layers + self.num_agg_layers)])
        self.ln2s = nn.ModuleList([nn.LayerNorm(embed_size) for _ in range(num_mha_layers + self.num_agg_layers)])
        self.mlps = nn.ModuleList([nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.Dropout(dropout),
            act(),
            nn.Linear(4 * embed_size, embed_size),
        ) for _ in range(num_mha_layers)])

        if self.include_agg_layer:
            self.mlps.append(
                nn.Sequential(
                    nn.Linear(embed_size, 4 * embed_size),
                    nn.Dropout(dropout),
                    act(),
                    nn.Linear(4 * embed_size, out_dim),
                )
            )
        else:
            self.mlps.append(
                nn.Sequential(
                    nn.Linear(embed_size, out_dim)
                )
            )

    def _split_heads(self, x: Tensor):
        batch_size, seq_len, _ = x.size()
        return (
            x.reshape(batch_size, seq_len, self.num_heads, self.dk)
            .transpose(1, 2)
            .reshape(batch_size * self.num_heads, seq_len, self.dk)
        )

    def _merge_heads(self, x: Tensor):
        batch_size, _, seq_len, _ = x.size()
        return (
            x.reshape(batch_size, self.num_heads, seq_len, self.dk)
            .transpose(1, 2)
            .reshape(batch_size, seq_len, self.dk * self.num_heads)
        )

    def forward(self, x: Tensor, avg_heads: bool = True, num_evals: int = 2):
        batch_size, seq_len, dim = x.shape
        threshold = 1 / seq_len
        batch_heads = batch_size * self.num_heads
        ladj, prior = 0, 0 
        flow_out, ladj = self.param_flow(x, num_evals=num_evals)
        path_matrix = torch.eye(self.seq_len, device=self.device).repeat(num_evals, batch_size, 1, 1).view(-1, seq_len, seq_len)
        attn_matrix = torch.eye(self.seq_len, device=self.device).repeat(num_evals, batch_size, 1, 1).view(-1, seq_len, seq_len)

        if self.hyper_type == "mask":
            mha_out, agg_out = torch.split(flow_out, split_size_or_sections=[self.total_mha_size, self.total_agg_size], dim=-1)
            mha_layers = torch.chunk(mha_out, chunks=self.num_mha_layers, dim=-1)
            agg_layers = torch.chunk(agg_out, chunks=self.num_agg_layers, dim=-1) if self.num_agg_layers > 0 else None
            for i in range(self.total_num_layers):
                agg_layer = False if i < self.num_mha_layers else True
                shape = 1 if agg_layer else self.seq_len
                mask_layer = agg_layers[i - self.num_mha_layers] if agg_layer else mha_layers[i]
                query = self.queries if agg_layer else None
                
                query_layer = self.query_layers[i]
                key_layer = self.key_layers[i]
                value_layer = self.value_layers[i]
                proj_layer = self.proj_layers[i]

                mha_func = partial(self._mha_mask, 
                                   mask_weights=mask_layer.view(-1, shape, self.seq_len),
                                   query_nn=query_layer,
                                   key_nn=key_layer,
                                   value_nn=value_layer,
                                   proj_nn=proj_layer,
                                   agg=agg_layer, 
                                   query=query,
                                   avg_heads=avg_heads,
                                   num_evals=num_evals)
                out, mask, adj = self._run_block(x, self.ln1s[i], self.ln2s[i], self.mlps[i], mha_func, agg_layer)
                thresh = (adj > threshold).float()
                attn_matrix = torch.bmm(thresh, attn_matrix)
                path_matrix = torch.bmm(mask, path_matrix)
                x = out

        elif self.hyper_type == "mha":
            mha_layers = torch.chunk(flow_out, chunks=self.total_num_layers, dim=-1)
            x = x.expand(num_evals, -1, -1, -1).reshape(-1, seq_len, dim)
            for i in range(self.total_num_layers):
                agg_layer = False if i < self.num_mha_layers else True
                query_layer, key_layer, value_layer, proj_layer  = torch.chunk(mha_layers[i], chunks=4, dim=-1)
                query = self.queries if agg_layer else None

                mha_func = partial(self._mha_mha, 
                                   Wq=query_layer.view(num_evals, self.embed_size, self.embed_size),
                                   Wk=key_layer.view(num_evals, self.embed_size, self.embed_size),
                                   Wv=value_layer.view(num_evals, self.embed_size, self.embed_size),
                                   Wo=proj_layer.view(num_evals, self.embed_size, self.embed_size),
                                   agg=agg_layer,
                                   query=query,
                                   avg_heads=avg_heads, 
                                   num_evals=num_evals)
                out, mask, adj = self._run_block(x, self.ln1s[i], self.ln2s[i], self.mlps[i], mha_func, agg_layer)
                thresh = (adj > threshold).float()
                attn_matrix = torch.bmm(thresh, attn_matrix)
                path_matrix = torch.bmm(mask, path_matrix)
                x = out # (b * e, l, k)
        
        elif self.hyper_type == "directa":
            mha_out, agg_out = torch.split(flow_out, split_size_or_sections=[self.total_mha_size, self.total_agg_size], dim=-1)
            mha_layers = torch.split(mha_out, self.num_mha_layers, dim=-1)
            agg_layers = torch.split(agg_out, self.num_agg_layers, dim=-1)
            for i in range(self.total_num_layers):
                agg_layer = False if i < self.num_mha_layers else True
                shape = 1 if agg_layer else self.seq_len
                value_layer = self.value_layers[i]
                proj_layer = self.proj_layers[i]
                attn_layer = agg_layers[i - self.num_mha_layers] if agg_layer else mha_layers[i]

                mha_func = partial(self._mha_directa, 
                                   logits=attn_layer.view(-1, shape, self.seq_len),
                                   value_nn=value_layer,
                                   proj_nn=proj_layer,
                                   agg=agg_layer,
                                   avg_heads=avg_heads, 
                                   num_evals=num_evals)
                out, mask, adj = self._run_block(x, self.ln1s[i], self.ln2s[i], self.mlps[i], mha_func, agg_layer)
                thresh = (adj > threshold).float()
                attn_matrix = torch.bmm(thresh, attn_matrix)
                path_matrix = torch.bmm(mask, path_matrix)
                x = out

        elif self.hyper_type == "qk":
            mha_layers = torch.chunk(flow_out, chunks=self.total_num_layers, dim=-1)
            x = x.expand(num_evals, -1, -1, -1).reshape(-1, seq_len, dim)
            for i in range(self.total_num_layers):
                agg_layer = False if i < self.num_mha_layers else True
                query_layer, key_layer  = torch.chunk(mha_layers[i], chunks=2, dim=-1)
                value_layer = self.value_layers[i]
                proj_layer = self.proj_layers[i]
                query = self.queries if agg_layer else None

                mha_func = partial(self._mha_qk, 
                                   Wq=query_layer.view(num_evals, self.embed_size, self.embed_size),
                                   Wk=key_layer.view(num_evals, self.embed_size, self.embed_size),
                                   value_nn=value_layer,
                                   proj_nn=proj_layer,
                                   agg=agg_layer,
                                   query=query,
                                   avg_heads=avg_heads, 
                                   num_evals=num_evals)
                
                out, mask, adj = self._run_block(x, self.ln1s[i], self.ln2s[i], self.mlps[i], mha_func, agg_layer)
                thresh = (adj > threshold).float()
                attn_matrix = torch.bmm(thresh, attn_matrix)
                path_matrix = torch.bmm(mask, path_matrix)
                x = out

        if self.training: 
            if self.prior_type == "laplace" and self.training:
                prior = self.prior().log_prob(path_matrix.sum(dim=(-2, -1)))
            elif self.prior_type == "normal" and self.training:
                prior = self.prior().log_prob(flow_out).sum(dim=-1)
            elif self.prior_type == "nf" and self.training:
                prior = self.prior().log_prob(flow_out)
            elif self.prior_type == "uniform" and self.training:
                prior = torch.tensor([1.0]).expand_as(ladj)

        if not self.include_agg_layer: 
            out = self.mlps[-1](out.max(dim=1)[0])
        else:
            out = out.squeeze(dim=1)

        return out, path_matrix, ladj, prior, attn_matrix

    @torch.inference_mode()
    def evaluate(self, x: Tensor, num_eval_samples: int = 5):
        batch_size, seq_len, hidden_dim = x.shape
        outs, masks, ladj, prior, attns = self(x, num_evals=num_eval_samples)
        outs = torch.sigmoid(outs)
        outs = outs.view(num_eval_samples, batch_size, -1)

        masks = masks.view(num_eval_samples, batch_size, -1, seq_len)
        attns = attns.view(num_eval_samples, batch_size, -1, seq_len)

        return outs.mean(dim=0), masks, attns

    def matmul(self, x: Tensor, W: Tensor, num_evals: int):
        batch_evals, seq_len, dim = x.shape
        return (x.view(num_evals, -1, seq_len, dim) @ W.unsqueeze(1)).view(-1, seq_len, dim)
    
    def _run_block(self, x: Tensor, ln1: nn.Module, ln2: nn.Module, mlp: nn.Module, mha_func, agg: bool = False):
        if self.layernorm:
            x_ln = ln1(x)
        else:
            x_ln = x
        attn_repr, mask, adj = mha_func(x_ln)
        if self.residual and not agg:
            attn_repr = attn_repr + x
            if self.layernorm:
                out = mlp(ln2(attn_repr))
            else:
                out = mlp(attn_repr)
            out = out + attn_repr
        else:
            if self.layernorm:
                out = mlp(ln2(attn_repr))
            else:
                out = mlp(attn_repr)
        return out, mask, adj
    
    def _mha_mask(
        self, 
        x: Tensor, 
        mask_weights: Tensor, 
        query_nn: nn.Module, 
        key_nn: nn.Module,
        value_nn: nn.Module,
        proj_nn: nn.Module,
        avg_heads: bool = True,
        agg: bool = False, 
        query: Tensor = None,
        bias: float = 0.5,
        num_evals: int = 1
    ):
        batch_size, seq_len, _ = x.size()
        shape = (1, seq_len) if agg else (seq_len, seq_len)
        queries = query_nn(x) if not agg else query_nn(query.expand(batch_size, -1, -1))
        keys = key_nn(x)
        values = value_nn(x)

        queries_split = self._split_heads(queries)  # (b * h, l, d_k)
        keys_split = self._split_heads(keys)
        values_split = self._split_heads(values)

        batch_heads = self.num_heads * batch_size
        edges_logit = mask_weights.view(batch_heads, -1) + bias
        edges_logit = torch.stack([torch.zeros_like(edges_logit), edges_logit], dim=-1)

        A = gumbel_softmax(edges_logit, tau=1.0, hard=True)
        A = A[:, :, -1].view(batch_heads, *shape)
        attention_logits = torch.matmul(queries_split, keys_split.transpose(-2, -1)) / math.sqrt(
            self.dk
        )
        attention_probs = F.softmax(attention_logits, dim=-1)
        masked_attention_probs = A * attention_probs
        hidden_repr = torch.matmul(masked_attention_probs, values_split)

        if self.residual and not agg:
            eye = torch.eye(seq_len, device=A.device).view(1, seq_len, seq_len).expand_as(A)
            A = A + eye 

        hidden_repr = hidden_repr.view(-1, self.num_heads, shape[0], self.dk)
        attention_repr = self._merge_heads(hidden_repr)
        attention_repr = proj_nn(attention_repr)

        if avg_heads:
            adjacency = masked_attention_probs.view(-1, self.num_heads, *shape).sum(dim=1)
            mask = A.view(-1, self.num_heads, shape[0], seq_len).sum(dim=1)
        else:
            adjacency = attention_probs
            mask = A.view(-1, self.num_heads, shape[0], seq_len)

        return attention_repr, mask, adjacency
    
    def _mha_mha(
        self, 
        x: Tensor, # (b * e, l, k)
        Wq: Tensor, 
        Wk: Tensor, 
        Wv: Tensor, 
        Wo: Tensor,
        agg: bool = False, 
        query: Tensor = None,
        avg_heads: bool = True,
        num_evals: int = 1
    ):
        batch_evals, seq_len, dim = x.shape
        shape = 1 if agg else seq_len
        if not agg:
            queries = self.matmul(x, Wq, num_evals=num_evals)
        else:
            queries = self.matmul(query.expand(batch_evals, -1, -1), Wq, num_evals=num_evals)
        keys = self.matmul(x, Wk, num_evals=num_evals)
        values = self.matmul(x, Wv, num_evals=num_evals) # (b*e, l, k)

        print(x.shape, keys.shape)

        queries_split = self._split_heads(queries)  
        keys_split = self._split_heads(keys)
        values_split = self._split_heads(values)

        attention_logits = torch.bmm(
            queries_split, keys_split.transpose(1, 2)
        ) / math.sqrt(
            self.dk
        )  # (b*h*e, l, l)

        attention_probs = softmax(attention_logits, dim=-1)
        hidden_repr = torch.bmm(attention_probs, values_split) 
        attention_repr = self._merge_heads(
            hidden_repr.view(-1, self.num_heads, shape, self.dk)
        ) # (b*e, l, k) 
        
        attention_repr = attention_repr.view(num_evals, -1, shape, dim) @ Wo.unsqueeze(1) # (b*e, l, k) @ (e, k, k)
        attention_repr = attention_repr.view(-1, shape, dim) 
        mask = torch.ones((batch_evals, self.num_heads, shape, seq_len), device=self.device)

        if avg_heads:
            adjacency = attention_probs.view(-1, self.num_heads, shape, seq_len).sum(dim=1)
            mask = mask.sum(dim=1)
        else:
            adjacency = attention_probs

        return attention_repr, mask, adjacency
    
    def _mha_directa(
        self, 
        x: Tensor, 
        logits: Tensor,
        value_nn: nn.Module,
        proj_nn: nn.Module,
        agg: bool = False, 
        avg_heads: bool = True,
        num_evals: int = 1
    ):
        batch_size, seq_len, _ = x.shape
        shape = 1 if agg else seq_len
        values = value_nn(x)
        values_split = self._split_heads(values)

        attention_probs = softmax(logits, dim=-1)
        hidden_repr = torch.bmm(attention_probs, values_split)
        attention_repr = self._merge_heads(
            hidden_repr.view(-1, self.num_heads, shape, self.dk)
        )

        attention_repr = proj_nn(attention_repr)
        mask = torch.ones((batch_size, self.num_heads, shape, seq_len))
        if avg_heads:
            adjacency = attention_probs.view(-1, self.num_heads, shape, seq_len).sum(dim=1)
            mask = mask.sum(dim=1)
        else:
            adjacency = attention_probs
        return attention_repr, mask, adjacency
    
    def _mha_qk(
        self, 
        x: Tensor, 
        Wq: Tensor, 
        Wk: Tensor,
        value_nn: nn.Module,
        proj_nn: nn.Module,
        agg: bool = False, 
        query: Tensor = None,
        avg_heads: bool = True,
        num_evals: int = 1
    ):
        batch_evals, seq_len, dim = x.shape
        shape = 1 if agg else seq_len
        if not agg:
            queries = self.matmul(x, Wq, num_evals=num_evals)
        else:
            queries = self.matmul(query.expand(batch_evals, -1, -1), Wq, num_evals=num_evals)
        keys = self.matmul(x, Wk, num_evals=num_evals)
        values = value_nn(x) 

        queries_split = self._split_heads(queries)  # (b * h * e, l, d_k)
        keys_split = self._split_heads(keys)
        values_split = self._split_heads(values)

        attention_logits = torch.bmm(
            queries_split, keys_split.transpose(1, 2)
        ) / math.sqrt(
            self.dk
        ) 

        attention_probs = softmax(attention_logits, dim=-1)
        hidden_repr = torch.bmm(attention_probs, values_split)

        attention_repr = self._merge_heads(
            hidden_repr.view(-1, self.num_heads, shape, self.dk)
        )
        attention_repr = proj_nn(attention_repr)
        mask = torch.ones((batch_evals, self.num_heads, shape, seq_len), device=self.device)

        if avg_heads:
            adjacency = attention_probs.view(-1, self.num_heads, shape, seq_len).sum(dim=1)
            mask = mask.sum(dim=1)
        else:
            adjacency = attention_probs

        return attention_repr, mask, adjacency

class HyperNetSpartan(nn.Module):

    def __init__(
        self, 
        inp_dim: int = 3,
        out_dim: int = 1, 
        prior_func = make_unit_gaussian,
        prior_type: str = "uniform",
        include_sparsity: bool = False,
        alpha: float = 0.1,
        num_mha_layers: int = 1,
        include_agg_layer: bool = False, 
        seq_len: int = 25,
        num_eval_samples: int = 5, 
        model_dim: int = 32, 
        num_heads: int = 1,
        dropout: float = 0.0,
        hyper_type: str = "qk",
        flow_params: dict = {"n_flows": 3, "hidden_features": [256, 256]},
        prior_params: dict = {"n_flows": 3, "hidden_features": (256, 256)},
        residual: bool = False,
        device: str = "cuda",
        forward_evals: int = 1, 
        layernorm: bool = True,
        separate_mask: bool = False,
        use_mask: bool = False,
        act: nn.Module = nn.ReLU,
        val_freq: int = 10, 
        force_vae_gaussian: bool = False,
        val_to_name: dict = {0: "id", 1: "col", 2: "pair", 3: "dist", 4: "comb"},
        pe: bool = True,
        sinusoidal: bool = True,
        embedding_inp: bool = True,
        lr: float = 1e-3,
        beta: float = 1.0,
        logger: WandbLogger = None,
        num_embeddings: int = 25,
        beta1: float = 0.9,
        beta2: float = 0.999,
        threshold: float = 0.01,
        *args, 
        **kwargs
    ):
        self.hyper_params = locals()
        
        for key in ["self", "__class__", "args", "kwargs"]:
            del self.hyper_params[key]

        super().__init__(*args, **kwargs)

        self.device = device
        self.logger = logger
        self.model_dim = model_dim
        self.val_freq = val_freq
        self.num_heads = num_heads
        self.num_mha_layers = num_mha_layers
        self.include_agg_layer = include_agg_layer
        self.num_eval_samples = num_eval_samples
        self.forward_evals = forward_evals

        if embedding_inp:
            self.embed_layer = nn.Embedding(num_embeddings, model_dim)

        bottleneck = 128
        self.feature_map = nn.Sequential(
            nn.Linear(model_dim if embedding_inp else inp_dim, bottleneck),
            act(),
            nn.Linear(bottleneck, model_dim),
            # nn.Identity()
        )

        if pe:
            if sinusoidal:
                model_dim = model_dim
            else:
                model_dim += 2
            embed_size = model_dim

        self.embed_size = embed_size
        self.pe = pe
        self.sinusoidal = sinusoidal
        self.embedding_inp = embedding_inp

        self.hyper_net = HyperNet(
            prior_func=prior_func,
            prior_type=prior_type,
            num_mha_layers=num_mha_layers,
            include_agg_layer=include_agg_layer,
            seq_len=seq_len,
            embed_size=embed_size,
            out_dim=out_dim,
            num_heads=num_heads,
            dropout=dropout,
            hyper_type=hyper_type,
            flow_params=flow_params,
            prior_params=prior_params,
            residual=residual,
            device=device,
            layernorm=layernorm,
            separate_mask=separate_mask,
            use_mask=use_mask, 
            act=act,
            force_vae_gaussian=force_vae_gaussian
        )

        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=lr, betas=(beta1, beta2)
        )
        self.accuracy = BinaryAccuracy()
        self.loss = nn.BCEWithLogitsLoss()
        self.global_step = 0
        self.threshold = threshold

        self.sparse_loss = L1SparsityAdjacency()
        self.alpha = alpha
        self.include_sparsity = include_sparsity
        self.max_paths = None
        self.val_to_name = val_to_name
        self.beta = beta

    def _enforce_sparsity(self, attns):
        num_edges = attns.sum(dim=(1, 2)) / self.max_paths
        return (self.alpha - num_edges).pow(2).mean()

    def forward(self, x: Tensor, evaluate: bool = False):
        priors, ladjs = 0, 0
        batch_size, width, height, _ = x.size()
        if self.max_paths is None:
            self.max_paths = compute_max_paths(
                width * height, self.num_heads, self.num_mha_layers, self.include_agg_layer
            )

            print(f"MAX PATHS: {self.max_paths}")

        self.threshold = 1 / (width * height)

        if self.embedding_inp:
            assert x.size(3) == 1, "channels is not 1 for shapes input"
            x = self.embed_layer(x.squeeze(3).int())  # (b, w, h, e)

        x_features = self.feature_map(x)
        if self.pe:
            if self.sinusoidal:
                embeddings = positionalencoding2d(
                    self.embed_size, height=height, width=width, device=self.device
                ).permute(  # returns (dim, h, w)
                    2, 1, 0
                )
                x_attn = x_features + embeddings.repeat(batch_size, 1, 1, 1)
                x_attn = x_attn.view(-1, width * height, self.embed_size)
            else:
                xs = torch.arange(width, device=self.device)
                ys = torch.arange(height, device=self.device)
                coords = torch.cartesian_prod(xs, ys).view(width, height, 2)
                coords = coords.expand(batch_size, width, height, 2)
                x_attn = torch.cat([x_features, coords], dim=-1)
                x_attn = x_attn.view(-1, width * height, self.embed_size)

        if evaluate:
            out, masks, attns = self.hyper_net.evaluate(x_attn, num_eval_samples=self.num_eval_samples)
            return out, masks, attns
        else:
            out, path_matrix, ladj, prior, attn_matrices = self.hyper_net(x_attn, num_evals=self.forward_evals)
            return out, path_matrix, ladj, prior, attn_matrices

        
    def fit(self, dataloader: DataLoader, num_epochs: int, testloaders: List):
        losses = []
        accs = []
        attn_edges = []
        mask_edges = []
        sparses = []
        gens = []

        attn_test = {i: [] for i in self.val_to_name.values()}
        masks_test = deepcopy(attn_test)
        losses_test = deepcopy(attn_test)
        accs_test = deepcopy(attn_test)

        postfix = {"loss": 0.0, "acc": 0.0, "gen_loss": 0.0}

        for step in (pbar := tqdm(range(1, num_epochs + 1))):
            self.train()
            epoch_loss = 0.0
            epoch_acc = 0.0
            epoch_sparse = 0.0
            epoch_gen = 0.0
            attn_running = 0.0
            mask_running = 0.0

            for batch_idx, batch in enumerate(dataloader):
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)
                out, masks, ladj, prior, attns = self(x)  # list of (b, l, l)
                gen_loss = (ladj - prior).mean()
                rec_loss = self.loss(out, y)
                epoch_gen += gen_loss.item()

                if self.include_sparsity:
                    sparse_loss = self._enforce_sparsity(masks)
                    epoch_sparse += sparse_loss.item()
                    loss = rec_loss + self.beta * gen_loss + sparse_loss
                else:
                    loss = rec_loss + self.beta * gen_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += rec_loss.item()
                with torch.no_grad():
                    acc = self.accuracy(out, y)
                    epoch_acc += acc.item()
                    
                    attn_running += compute_mask_mean(attns)
                    mask_running += compute_mask_mean(masks)

                self.global_step += 1

            epoch_loss /= len(dataloader)
            epoch_acc /= len(dataloader)
            epoch_sparse /= len(dataloader)
            epoch_gen /= len(dataloader)
            attn_running /= len(dataloader)
            mask_running /= len(dataloader)

            losses.append(epoch_loss)
            accs.append(epoch_acc)
            sparses.append(epoch_sparse)
            gens.append(epoch_gen)
            attn_edges.append(attn_running)
            mask_edges.append(mask_running)

            postfix["loss"] = epoch_loss
            postfix["acc"] = epoch_acc
            postfix["gen_loss"] = epoch_loss

            pbar.set_description(f"Epoch: {step}")
            self.logger.log_metrics({"train/loss_epoch": epoch_loss}, step=step)
            self.logger.log_metrics({"train/acc_epoch": epoch_acc}, step=step)

            if self.include_sparsity:
                self.logger.log_metrics({"train/sparse_loss": epoch_sparse}, step=step)
                postfix["sparse_loss"] = epoch_sparse

            self.logger.log_metrics(
                {f"train/attn_edges_train": attn_running}, step=self.global_step
            )

            self.logger.log_metrics(
                {f"train/mask_edges_train": mask_running}, step=self.global_step
            )

            if step % self.val_freq == 0: 
                for loader, name in zip(testloaders, self.val_to_name.values()):
                    test_metrics = self.test(name, loader, folder="val")
                    if "id" in name:
                        postfix["val_id"] = test_metrics["acc"]
                    elif "a" in name:
                        postfix["val_a"] = test_metrics["acc"]
                    elif "b" in name:
                        postfix["val_b"] = test_metrics["acc"]
                    masks_test[name].append(test_metrics["mask"])
                    attn_test[name].append(test_metrics["attn"])
                    losses_test[name].append(test_metrics["loss"])
                    accs_test[name].append(test_metrics["acc"])

            postfix["mask_edges"] = mask_running
            # postfix["attn_edges"] = attn_running

            # if self.agg_pool:
            #     self.out.temp_decay(step, num_epochs)
            #     postfix["temp"] = self.out.temp

            pbar.set_postfix(postfix)

        return (
            losses,
            accs,
            sparses,
            gens,
            mask_edges,
            attn_edges,
            losses_test,
            accs_test,
            attn_test,
            masks_test,
        )

    def test(self, name: str, dataloader: DataLoader, folder: str = "test"):
        self.eval()
        attn_running = 0.0
        mask_running = 0.0
        epoch_acc = 0.0
        epoch_loss = 0.0

        for batch_idx, batch in enumerate(dataloader):
            x, y = batch
            x = x.to(self.device)
            y = y.to(self.device)
            out, masks, attns = self(x, evaluate=True)
            loss = self.loss(out, y)

            epoch_loss += loss.item()
            with torch.no_grad():
                acc = self.accuracy(out, y)
                epoch_acc += acc.item()
                attn_running += compute_mask_mean(attns)
                mask_running += compute_mask_mean(masks)

        epoch_loss /= len(dataloader)
        epoch_acc /= len(dataloader)
        attn_running /= len(dataloader)
        mask_running /= len(dataloader)

        self.logger.log_metrics(
            {f"{folder}/loss_epoch_{name}": epoch_loss}, step=self.global_step
        )

        self.logger.log_metrics(
            {f"{folder}/acc_epoch_{name}": epoch_acc}, step=self.global_step
        )

        self.logger.log_metrics(
            {f"{folder}/attn_edges_{name}": attn_running}, step=self.global_step
        )

        self.logger.log_metrics(
            {f"{folder}/mask_edges_{name}": mask_running}, step=self.global_step
        )

        self.train()

        return {
            "loss": epoch_loss,
            "acc": epoch_acc,
            "attn": attn_running,
            "mask": mask_running,
        }

    @torch.no_grad()
    def test_anti(self, anti_dataset: DataLoader):
        self.eval()
        # total acc, acc a, acc b, conf a, conf b
        results = {}
        labels = []
        true_labels = []
        for batch_idx, (x, y) in enumerate(anti_dataset):
            x = x.to(self.device)
            y = y.to(self.device)
            probs, masks, attns = self(x, evaluate=True)
            labels.append(probs)
            true_labels.append(y)

        preds = torch.cat(labels, dim=0)
        trues = torch.cat(true_labels, dim=0)
        size = preds.size(0)
        midpoint = size // 2

        total_acc = self.accuracy(preds, trues)
        results["total_acc"] = total_acc.item()

        acc_a = self.accuracy(preds[:midpoint], trues[:midpoint])
        acc_b = self.accuracy(preds[midpoint:], trues[midpoint:])
        conf_a = preds[:midpoint].mean()
        conf_b = preds[:midpoint].mean()

        results["acc_a"] = acc_a.item()
        results["acc_b"] = acc_b.item()
        results["conf_a"] = conf_a.item()
        results["conf_b"] = conf_b.item()

        self.train()
        return results