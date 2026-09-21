import math
import torch
import torch.nn as nn
import zuko

from functools import partial
from torch import Tensor
from torch.nn.functional import softmax, gumbel_softmax

from sparse_generalization.layers.priors import LaplacePrior, NormalPrior, make_unit_gaussian
from sparse_generalization.layers.vae import FlowVAE
from sparse_generalization.layers.vhypernet import VHyperNet
from sparse_generalization.utils.util_funcs import get_device, resolve_train_query, register_query_grad_ema, with_residual_edges
from sparse_generalization.losses.criterion import Criterion
from sparse_generalization.layers.diversity_losses import CosineRepDiv, L2DistanceDiv


class HyperNet(nn.Module):

    def __init__(
        self,
        weight_gen = partial(FlowVAE, prior_func=make_unit_gaussian),  # partial of FlowVAE | VHyperNet
        prior_type: str = "uniform",
        num_mha_layers: int = 1,
        include_agg_layer: bool = False,
        seq_len: int = 1,
        embed_size: int = 32,
        out_dim: int = 2,
        criterion: Criterion = Criterion("ce"),  # shared with HyperNetSpartan
        num_heads: int = 1,
        dropout: float = 0.0,
        hyper_type: str = "qk",  
        prior_params: dict = {"n_flows": 3, "hidden_features": (256, 256)},
        residual: bool = False,
        div_loss: nn.Module = CosineRepDiv,
        device: str | None = None,
        layernorm: bool = True,
        separate_mask: bool = False,
        use_mask: bool = False,
        act: nn.Module = nn.ReLU,
        train_query: str = "train",  # 'fixed' | 'train' | 'ema'
        agg_ema: float = 0.99,  # ema coefficient of the agg query's gradient; only used when train_query == 'ema'
        forward_evals: int = 1,
        *args,
        **kwargs,
    ):
        device = get_device(device)
        super().__init__(*args, **kwargs)
        if embed_size % num_heads != 0:
            raise SyntaxError(
                f"Embed Size not divisible by number of heads, embed_size % num_heads = {embed_size % num_heads}"
            )
        if hyper_type not in ("mask", "mha", "qk"):
            raise ValueError(f"hyper_type must be 'mask', 'mha' or 'qk', got {hyper_type!r}")

        self.hyper_type = hyper_type
        self.num_heads = num_heads
        self.residual = residual
        self.criterion = criterion
        self.prior_type = prior_type
        self.seq_len = seq_len
        self.div_loss = div_loss()
        self.num_mha_layers = num_mha_layers
        self.include_agg_layer = include_agg_layer
        self.device = device
        self.embed_size = embed_size
        self.layernorm = layernorm
        self.dk = embed_size // num_heads
        self.num_agg_layers = 1 if include_agg_layer else 0
        self.total_num_layers = self.num_agg_layers + self.num_mha_layers

        def linears():
            return nn.ModuleList([nn.Linear(embed_size, embed_size) for _ in range(self.total_num_layers)])

        if hyper_type == "mask":
            self.query_layers, self.key_layers = linears(), linears()
            self.value_layers, self.proj_layers = linears(), linears()
            self.base_dist_size, self.agg_dist_size = seq_len ** 2, seq_len
            use_encoder = encoder_heads = True
        elif hyper_type == "mha":
            self.base_dist_size = 4 * embed_size ** 2
            self.agg_dist_size = self.base_dist_size
            use_encoder = encoder_heads = False
        elif hyper_type == "qk":
            self.value_layers, self.proj_layers = linears(), linears()
            self.base_dist_size = 2 * embed_size ** 2
            self.agg_dist_size = self.base_dist_size
            use_encoder = encoder_heads = False

        self.train_query = resolve_train_query(train_query)
        self.agg_ema = agg_ema
        self.queries = nn.init.uniform_(
            nn.Parameter(torch.zeros((1, embed_size), device=device), requires_grad=self.train_query != "fixed")
        )
        if self.train_query == "ema":
            register_query_grad_ema(self, "queries", agg_ema)
        self.total_mha_size = self.num_mha_layers * self.base_dist_size
        self.total_agg_size = self.num_agg_layers * self.agg_dist_size
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

        # generator-specific args (flow_params, prior_func, hidden_features, ...) live in the partial
        self.param_flow = weight_gen(
            input_dim=embed_size,
            output_dim=self.total_dist_size,
            num_heads=num_heads,
            encoder_heads=encoder_heads,
            use_encoder=use_encoder,
            use_mask=use_mask,
            separate_mask=separate_mask,
            layernorm=layernorm,
            train_query=train_query,
            agg_ema=agg_ema,
            act=act,
            num_modes=forward_evals,
            device=device,
        )
        self.fixed_evals = isinstance(self.param_flow, VHyperNet)
        self.forward_evals = forward_evals

        self.ln1s = nn.ModuleList([nn.LayerNorm(embed_size) for _ in range(self.total_num_layers)])
        self.ln2s = nn.ModuleList([nn.LayerNorm(embed_size) for _ in range(self.total_num_layers)])
        self.mlps = nn.ModuleList([nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.Dropout(dropout),
            act(),
            nn.Linear(4 * embed_size, embed_size),
        ) for _ in range(num_mha_layers)])

        # final layer: either an aggregation block or a plain linear head over max-pooled tokens
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
            self.mlps.append(nn.Sequential(nn.Linear(embed_size, out_dim)))

    def effective_evals(self, num_evals: int):
        # the mode hypernet always produces exactly one weight set per mode (= forward_evals)
        return self.forward_evals if self.fixed_evals else num_evals

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

    def _layer_weights(self, flow_out: Tensor):
        if self.hyper_type != "mask":
            return torch.chunk(flow_out, chunks=self.total_num_layers, dim=-1)  # (e, size)
        # mask weights are generated per head: (e * b, h * D) -> (e * b, h, D)
        flow_out = flow_out.view(flow_out.size(0), self.num_heads, self.total_dist_size)
        mha_out, agg_out = torch.split(flow_out, [self.total_mha_size, self.total_agg_size], dim=-1)
        layers = list(torch.chunk(mha_out, chunks=self.num_mha_layers, dim=-1))
        if self.include_agg_layer:
            layers.append(agg_out)
        return layers

    def _mha_func(self, i: int, weights: Tensor, agg: bool, avg_heads: bool, num_evals: int):
        query = self.queries if agg else None
        shape = 1 if agg else self.seq_len
        common = dict(agg=agg, avg_heads=avg_heads, num_evals=num_evals)
        view_w = lambda w: w.view(num_evals, self.embed_size, self.embed_size)

        if self.hyper_type == "mask":
            return partial(
                self._mha_mask,
                mask_weights=weights.reshape(-1, shape, self.seq_len),  # (e * b * h, shape, l)
                query_nn=self.query_layers[i],
                key_nn=self.key_layers[i],
                value_nn=self.value_layers[i],
                proj_nn=self.proj_layers[i],
                query=query,
                **common,
            )
        if self.hyper_type == "mha":
            Wq, Wk, Wv, Wo = torch.chunk(weights, chunks=4, dim=-1)
            return partial(
                self._mha_mha,
                Wq=view_w(Wq), Wk=view_w(Wk), Wv=view_w(Wv), Wo=view_w(Wo),
                query=query,
                **common,
            )
        # qk
        Wq, Wk = torch.chunk(weights, chunks=2, dim=-1)
        return partial(
            self._mha_qk,
            Wq=view_w(Wq), Wk=view_w(Wk),
            value_nn=self.value_layers[i],
            proj_nn=self.proj_layers[i],
            query=query,
            **common,
        )

    def forward(self, x: Tensor, avg_heads: bool = True, num_evals: int = 2, compute_div: bool = False):
        batch_size, seq_len, dim = x.shape
        num_evals = self.effective_evals(num_evals)
        threshold = 1 / seq_len
        prior = 0
        div = torch.tensor([0.0], device=x.device)
        flow_out, ladj = self.param_flow(x, num_evals=num_evals)
        layer_weights = self._layer_weights(flow_out)

        attn_maps = []
        # layers return per-sample (e * b) masks when heads are averaged, per-head (e * b * h) otherwise
        mask_batch = num_evals * batch_size * (1 if avg_heads else self.num_heads)
        eye = torch.eye(self.seq_len, device=self.device)
        path_matrix = eye.expand(mask_batch, seq_len, seq_len).clone()
        attn_matrix = path_matrix.clone()

        # (b, l, d) -> (e * b, l, d): one copy of the input per sampled weight set
        x = x.expand(num_evals, -1, -1, -1).reshape(-1, seq_len, dim)

        for i in range(self.total_num_layers):
            agg_layer = i >= self.num_mha_layers
            mha_func = self._mha_func(i, layer_weights[i], agg_layer, avg_heads, num_evals)
            out, mask, adj = self._run_block(x, self.ln1s[i], self.ln2s[i], self.mlps[i], mha_func, agg_layer)
            attn_maps.append(adj)  # (e * b [* h], l, l)
            edges = with_residual_edges((adj > threshold).float(), self.residual)
            attn_matrix = torch.bmm(edges, attn_matrix)
            path_matrix = torch.bmm(mask, path_matrix)
            x = out

        if self.training:
            if self.prior_type == "laplace":
                num_paths = path_matrix.sum(dim=(-2, -1)).view(num_evals, batch_size, -1).sum(dim=-1)
                if ladj.size(0) == num_evals:
                    num_paths = num_paths.mean(dim=1)
                prior = self.prior().log_prob(num_paths.reshape(-1))
            elif self.prior_type == "normal":
                prior = self.prior().log_prob(flow_out).sum(dim=-1)
            elif self.prior_type == "nf":
                prior = self.prior().log_prob(flow_out.reshape(-1, self.total_dist_size))
                prior = prior.view(flow_out.size(0), -1).sum(dim=-1)
            elif self.prior_type == "uniform":
                prior = torch.ones_like(ladj)

        if compute_div:
            match self.div_loss:
                case CosineRepDiv() | L2DistanceDiv():
                    div = self.div_loss(out.reshape(num_evals, batch_size, out.size(-2), -1))

        if self.include_agg_layer:
            out = out.squeeze(dim=1)
        else:
            out = self.mlps[-1](out.max(dim=1)[0])

        return out, path_matrix, ladj, prior, attn_matrix, div

    @torch.inference_mode()
    def evaluate(self, x: Tensor, num_eval_samples: int = 5, ret_mean: bool = True):
        batch_size, seq_len, _ = x.shape
        num_eval_samples = self.effective_evals(num_eval_samples)
        outs, masks, _, _, attns, _ = self(x, num_evals=num_eval_samples)
        outs = self.criterion.probs(outs).view(num_eval_samples, batch_size, -1)  # (e, b, c) class probabilities
        masks = masks.view(num_eval_samples, batch_size, -1, seq_len)
        attns = attns.view(num_eval_samples, batch_size, -1, seq_len)

        if ret_mean:
            return outs.mean(dim=0), masks, attns
        return outs, masks, attns

    def matmul(self, x: Tensor, W: Tensor, num_evals: int):
        _, seq_len, dim = x.shape
        return (x.view(num_evals, -1, seq_len, dim) @ W.unsqueeze(1)).view(-1, seq_len, dim)

    def _run_block(self, x: Tensor, ln1: nn.Module, ln2: nn.Module, mlp: nn.Module, mha_func, agg: bool = False):
        ln1, ln2 = (ln1, ln2) if self.layernorm else (nn.Identity(), nn.Identity())
        attn_repr, mask, adj = mha_func(ln1(x))
        if self.residual and not agg:
            attn_repr = attn_repr + x
            out = mlp(ln2(attn_repr)) + attn_repr
        else:
            out = mlp(ln2(attn_repr))
        return out, mask, adj

    def _mask_and_adjacency(self, attention_probs: Tensor, mask: Tensor, shape: int, seq_len: int, avg_heads: bool):
        if avg_heads:
            adjacency = attention_probs.view(-1, self.num_heads, shape, seq_len).sum(dim=1)
            mask = mask.view(-1, self.num_heads, shape, seq_len).sum(dim=1)
        else:
            adjacency = attention_probs
            mask = mask.view(-1, shape, seq_len)
        return mask, adjacency

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
        batch_evals, seq_len, _ = x.size()
        shape = 1 if agg else seq_len
        queries = query_nn(query.expand(batch_evals, -1, -1)) if agg else query_nn(x)
        keys = key_nn(x)
        values = value_nn(x)

        queries_split = self._split_heads(queries)  # (b * h * e, l, d_k)
        keys_split = self._split_heads(keys)
        values_split = self._split_heads(values)

        batch_heads = self.num_heads * batch_evals
        edges_logit = mask_weights.view(batch_heads, -1) + bias
        edges_logit = torch.stack([torch.zeros_like(edges_logit), edges_logit], dim=-1)
        A = gumbel_softmax(edges_logit, tau=1.0, hard=True)[:, :, -1].view(batch_heads, shape, seq_len)

        attention_logits = torch.bmm(queries_split, keys_split.transpose(-2, -1)) / math.sqrt(self.dk)
        attention_probs = softmax(attention_logits, dim=-1)
        masked_attention_probs = A * attention_probs
        hidden_repr = torch.bmm(masked_attention_probs, values_split)

        if self.residual and not agg:
            A = A + torch.eye(seq_len, device=A.device).expand_as(A)

        attention_repr = self._merge_heads(hidden_repr.view(-1, self.num_heads, shape, self.dk))
        attention_repr = proj_nn(attention_repr)

        adj_probs = masked_attention_probs if avg_heads else attention_probs
        mask, adjacency = self._mask_and_adjacency(adj_probs, A, shape, seq_len, avg_heads)

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
        q_inp = query.expand(batch_evals, -1, -1) if agg else x
        queries = self.matmul(q_inp, Wq, num_evals=num_evals)
        keys = self.matmul(x, Wk, num_evals=num_evals)
        values = self.matmul(x, Wv, num_evals=num_evals) # (b*e, l, k)

        queries_split = self._split_heads(queries)
        keys_split = self._split_heads(keys)
        values_split = self._split_heads(values)

        attention_logits = torch.bmm(queries_split, keys_split.transpose(1, 2)) / math.sqrt(self.dk)  # (b*h*e, l, l)
        attention_probs = softmax(attention_logits, dim=-1)
        hidden_repr = torch.bmm(attention_probs, values_split)
        attention_repr = self._merge_heads(hidden_repr.view(-1, self.num_heads, shape, self.dk))  # (b*e, l, k)
        attention_repr = self.matmul(attention_repr, Wo, num_evals=num_evals).view(-1, shape, dim)

        mask = torch.ones((batch_evals, self.num_heads, shape, seq_len), device=self.device)
        mask, adjacency = self._mask_and_adjacency(attention_probs, mask, shape, seq_len, avg_heads)

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
        batch_evals, seq_len, _ = x.shape
        shape = 1 if agg else seq_len
        q_inp = query.expand(batch_evals, -1, -1) if agg else x
        queries = self.matmul(q_inp, Wq, num_evals=num_evals)
        keys = self.matmul(x, Wk, num_evals=num_evals)
        values = value_nn(x)

        queries_split = self._split_heads(queries)  # (b * h * e, l, d_k)
        keys_split = self._split_heads(keys)
        values_split = self._split_heads(values)

        attention_logits = torch.bmm(queries_split, keys_split.transpose(1, 2)) / math.sqrt(self.dk)
        attention_probs = softmax(attention_logits, dim=-1)
        hidden_repr = torch.bmm(attention_probs, values_split)
        attention_repr = self._merge_heads(hidden_repr.view(-1, self.num_heads, shape, self.dk))
        attention_repr = proj_nn(attention_repr)

        mask = torch.ones((batch_evals, self.num_heads, shape, seq_len), device=self.device)
        mask, adjacency = self._mask_and_adjacency(attention_probs, mask, shape, seq_len, avg_heads)

        return attention_repr, mask, adjacency
