#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path


def main():
    p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,required=True); a=p.parse_args()
    s=a.root/"project/stage6_5_calibrated_v3"; maker=s/"runner/make_status.py"
    while True:
        subprocess.run(["python",str(maker),"--root",str(a.root)],check=False)
        if (s/"analysis/manifest.json").exists(): break
        pid_path=s/"logs/pipeline.pid"
        try: pid=int(pid_path.read_text().strip())
        except Exception: pid=None
        if pid:
            try: Path(f"/proc/{pid}").stat()
            except OSError: break
        time.sleep(10)


if __name__=="__main__": main()
