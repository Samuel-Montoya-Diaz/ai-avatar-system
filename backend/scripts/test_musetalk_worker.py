#!/usr/bin/env python
"""
Smoke test for the MuseTalk worker, outside FastAPI.

    docker compose exec backend python scripts/test_musetalk_worker.py \
        --image /opt/MuseTalk/assets/demo/man/man.png --audio /tmp/test.wav

Starts the worker the way app/services/animator.py does, then reports model
load time, time per job and peak GPU memory. Job 1 includes face detection;
later jobs reuse the cached coordinates, so they show the steady-state cost.
Peak VRAM is for the whole GPU (it includes other processes such as Ollama).
"""
import argparse
import json
import os
import subprocess
import threading
import time
from pathlib import Path

MT = Path(os.environ.get("MUSETALK_PATH", "/opt/MuseTalk"))
PY = os.environ.get("MUSETALK_PYTHON") or "python"
WORKER = Path(__file__).resolve().parent.parent / "musetalk_worker.py"


def vram_used_mib() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    ).stdout.strip().splitlines()
    return int(out[0])


class Peak(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.peak, self.stop = vram_used_mib(), False

    def run(self):
        while not self.stop:
            self.peak = max(self.peak, vram_used_mib())
            time.sleep(0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--out", default="/tmp/mt_out")
    a = ap.parse_args()

    Path(a.out).mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(MT)
    baseline = vram_used_mib()
    peak = Peak()
    peak.start()

    stderr = open(Path(a.out) / "worker_stderr.log", "wb")
    t0 = time.time()
    p = subprocess.Popen([PY, str(WORKER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=stderr, cwd=str(MT), env=env, text=True)
    init = {
        "unet_model_path": str(MT / "models/musetalkV15/unet.pth"),
        "unet_config": str(MT / "models/musetalkV15/musetalk.json"),
        "whisper_dir": str(MT / "models/whisper"),
        "vae_type": str(MT / "models/sd-vae"),
        "use_float16": True,
        "face_restore": "off",
        "face_restore_model": "",
    }
    p.stdin.write(json.dumps(init) + "\n")
    p.stdin.flush()
    line = p.stdout.readline().strip()
    print(f"worker: {line!r} after {time.time() - t0:.1f}s")
    if line != "READY":
        print("FAILED to start; see", Path(a.out) / "worker_stderr.log")
        return 1
    print(f"VRAM after load: {vram_used_mib()} MiB (baseline {baseline})")

    for i in range(1, a.runs + 1):
        out = str(Path(a.out) / f"out_{i}.mp4")
        job = {"image": a.image, "audio": a.audio, "output": out,
               "coord_cache": str(Path(a.out) / "coords.pkl")}
        t = time.time()
        p.stdin.write(json.dumps(job) + "\n")
        p.stdin.flush()
        res = json.loads(p.stdout.readline())
        print(f"job {i}: {res['status']} in {time.time() - t:.1f}s -> {out}")
        if res["status"] != "ok":
            print(res.get("msg"), res.get("tb", ""))

    peak.stop = True
    print(f"VRAM peak: {peak.peak} MiB (baseline {baseline}, delta {peak.peak - baseline})")
    p.stdin.close()
    p.wait(timeout=30)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
