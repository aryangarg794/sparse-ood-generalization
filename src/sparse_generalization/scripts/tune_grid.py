import itertools
import json
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import dill
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
NAME = "tune_grid32"
OUT = ROOT / "results" / NAME
SUMMARY = OUT / "summary.jsonl"
SEEDS = [0, 1, 2]
WORKERS = 5
THRESH = 0.95
FIT = 0.99
MAX_FAILURES = 3

GRID = {
    "lr": [3e-5, 1e-4, 3e-4],
    "qkscale": [4, 6, 8],
    "eta_div_target": [0.05, 0.15],
    "eta_div_coef": [1, 2],
    "z_dim": [8, 16],
    "zero_last_bias": [False, True],
}

lock = threading.Lock()
failures = []


def cell_id(p: dict) -> str:
    return "lr{lr:g}_qk{qkscale:g}_t{eta_div_target:g}_c{eta_div_coef:g}_z{z_dim}_zb{b}".format(
        b=int(p["zero_last_bias"]), **p
    )


def cells():
    keys = list(GRID)
    for vals in itertools.product(*GRID.values()):
        yield dict(zip(keys, vals))


def result_path(cid: str, seed: int) -> Path:
    return OUT / f"{cid}_s{seed}.pl"


def overrides(p: dict, cid: str, seed: int) -> list:
    return [
        "model=hypernet_unit",
        "model.weight_gen.hidden_features=[32,32]",
        "data.data_func.size=10000",
        f"seeds=[{seed}]",
        "num_concurrent=1",
        f"run_name={NAME}/{cid}_s{seed}",
        f"model.lr={p['lr']}",
        f"model.qkscale={p['qkscale']}",
        f"model.eta_div_target={p['eta_div_target']}",
        f"model.eta_div_coef={p['eta_div_coef']}",
        f"model.weight_gen.z_dim={p['z_dim']}",
        f"model.weight_gen.zero_last_bias={str(p['zero_last_bias']).lower()}",
    ]


def seed_metrics(cid: str, seed: int) -> dict:
    with open(result_path(cid, seed), "rb") as f:
        r = dill.load(f)[seed]
    a = np.array(r["test_a"]["mode_accs"])
    b = np.array(r["test_b"]["mode_accs"])
    split = bool((((a >= THRESH) & (b < THRESH)).any()) and (((b >= THRESH) & (a < THRESH)).any()))
    id_acc = float(r["test_id"]["acc"])
    n = len(a)
    pair = (a[:, None] + b[None, :]) / 2
    np.fill_diagonal(pair, -np.inf)
    return {
        "seed": seed,
        "separated": split,
        "fit": id_acc >= FIT,
        "success": split and id_acc >= FIT,
        "id_acc": id_acc,
        "split_score": float(pair.max()) if n > 1 else float("nan"),
        "a_max": float(a.max()),
        "b_max": float(b.max()),
    }


def summarize(p: dict, cid: str):
    per_seed = [seed_metrics(cid, s) for s in SEEDS]
    row = {
        "cell": cid,
        "params": p,
        "score": int(sum(m["success"] for m in per_seed)),
        "n_seeds": len(per_seed),
        "n_separated": int(sum(m["separated"] for m in per_seed)),
        "mean_split_score": float(np.mean([m["split_score"] for m in per_seed])),
        "seeds": per_seed,
    }
    with lock, open(SUMMARY, "a") as f:
        f.write(json.dumps(row) + "\n")


def run_job(job):
    p, cid, seed = job
    if result_path(cid, seed).exists() or len(failures) >= MAX_FAILURES:
        return
    log = OUT / "logs" / f"{cid}_s{seed}.log"
    cmd = ["uv", "run", "train"] + overrides(p, cid, seed)
    for attempt in range(2):
        with open(log, "w") as lf:
            code = subprocess.call(cmd, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT)
        if code == 0 and result_path(cid, seed).exists():
            return
    with lock:
        failures.append(f"{cid}_s{seed} (log: {log})")


def done_cells() -> set:
    if not SUMMARY.exists():
        return set()
    return {json.loads(line)["cell"] for line in SUMMARY.read_text().splitlines() if line}


def main():
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    finished = done_cells()
    jobs = [(p, cell_id(p), s) for p in cells() if cell_id(p) not in finished for s in SEEDS]
    pending = {}
    for p, cid, s in jobs:
        pending.setdefault(cid, [p, 0])[1] += 1

    def work(job):
        run_job(job)
        p, cid, s = job
        with lock:
            pending[cid][1] -= 1
            complete = pending[cid][1] == 0
        if complete and all(result_path(cid, x).exists() for x in SEEDS):
            summarize(p, cid)

    with ThreadPoolExecutor(WORKERS) as ex:
        list(ex.map(work, jobs))

    if failures:
        print("FAILED JOBS:\n" + "\n".join(failures))
        sys.exit(1)
    print(f"DONE: {len(done_cells())} cells -> {SUMMARY}")


if __name__ == "__main__":
    main()
