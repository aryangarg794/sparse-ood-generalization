# Sparse Transformers for OOD Generalization 
---------

## Setup & Running

Download [uv](https://docs.astral.sh/uv/getting-started/installation/) and then run 
```bash
uv sync
```

This downloads all the libraries etc. 

### Scripts

The repo uses hydra alongside uv. To run a script just run 
```bash
uv run [SCRIPT] [ARGS]
```

The scripts can be found in the scripts folder under src/sparse_generalization or in the pyproject.toml. Basic python scripting also works. 

Since we use hydra-core, the repo runs scripts with models based on the config files (`src/sparse_generalization/config`). Then to change parameters in the config we can use CLI args such as 

```bash
uv run shapes model.agg_pool=true
```
where model refers to the model defined in the `default.yaml`. To change model we can do something like `uv run shapes model=your_model` where your_model is defined as a yaml file under `model/`.

More info about hydra: [link](https://hydra.cc/docs/intro/). 