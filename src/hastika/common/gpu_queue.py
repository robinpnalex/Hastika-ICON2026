"""Run shell jobs across the GPUs of one machine, one job per GPU at a time.

Kaggle's T4 x2 is two separate 15 GB cards, not one 30 GB pool, so the useful
parallelism is "a different job on each card". Each job is pinned with
CUDA_VISIBLE_DEVICES, logs to its own file, and the queue prints the last line of every
running job once a minute so a Kaggle cell shows progress without interleaved output.
"""
import os
import subprocess
import sys
import time
from collections import deque


def _tail(path):
    try:
        with open(path, "rb") as fh:
            fh.seek(0, 2)
            fh.seek(max(0, fh.tell() - 2000))
            lines = fh.read().decode("utf-8", "replace").strip().splitlines()
        return lines[-1][:160] if lines else ""
    except OSError:
        return ""


def run_queue(jobs, n_gpus=None, poll=60, raise_on_fail=True):
    """jobs: list of dicts with name, cmd (str, run with bash), log, optional env/cwd.
    Returns {name: returncode}."""
    import torch
    n_gpus = n_gpus or max(1, torch.cuda.device_count())
    todo, running, codes = deque(jobs), {}, {}
    free = list(range(n_gpus))
    t0 = time.time()
    while todo or running:
        while todo and free:
            job, gpu = todo.popleft(), free.pop(0)
            env = {**os.environ, **job.get("env", {}), "CUDA_VISIBLE_DEVICES": str(gpu)}
            fh = open(job["log"], "w")
            p = subprocess.Popen(["bash", "-c", job["cmd"]], stdout=fh, stderr=subprocess.STDOUT,
                                 env=env, cwd=job.get("cwd"))
            running[job["name"]] = (p, gpu, fh, job["log"])
            print(f"[{(time.time() - t0) / 60:6.1f} min] start {job['name']} on GPU {gpu}",
                  flush=True)
        time.sleep(poll)
        for name, (p, gpu, fh, log) in list(running.items()):
            if p.poll() is None:
                print(f"[{(time.time() - t0) / 60:6.1f} min] {name:24s} {_tail(log)}", flush=True)
                continue
            fh.close()
            codes[name] = p.returncode
            free.append(gpu)
            del running[name]
            state = "done" if p.returncode == 0 else f"FAILED ({p.returncode})"
            print(f"[{(time.time() - t0) / 60:6.1f} min] {state} {name}; log {log}", flush=True)
            if p.returncode:
                with open(log, errors="replace") as fh2:
                    sys.stdout.write("".join(fh2.readlines()[-40:]))
    failed = [n for n, c in codes.items() if c]
    if failed and raise_on_fail:
        raise RuntimeError(f"jobs failed: {failed}")
    return codes
