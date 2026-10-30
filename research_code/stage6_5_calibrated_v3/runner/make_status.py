#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path


def alive(pid):
    if not pid: return False
    try: os.kill(int(pid),0); return True
    except (OSError,ValueError): return False


def read_pid(path):
    try: return int(path.read_text().strip())
    except Exception: return None


def main():
    p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,required=True); p.add_argument("--phase"); p.add_argument("--state")
    a=p.parse_args(); s=a.root/"project/stage6_5_calibrated_v3"; target=s/"status.json"
    old={}
    if target.exists():
        try: old=json.loads(target.read_text(encoding="utf-8"))
        except Exception: pass
    pid=read_pid(s/"logs/pipeline.pid"); running=alive(pid)
    enc=sum((s/"runs"/name).joinpath("checkpoints/best.pt").exists() for name in ("lora_encoder","partial_last2_encoder"))
    adapters=sum((s/"runs"/name).joinpath("patch_checkpoints/best.pt").exists() for name in ("control_adapter","full_adapter"))
    captions=sum((s/"caption_results"/v/"checkpoints"/f"{v}_best.pt").exists() for v in ("raw","control","full","lora","partial_last2"))
    metrics=len(list((s/"evaluation/results").glob("*/*/metrics.json"))) if (s/"evaluation/results").exists() else 0
    baseline=len(list((s/"baseline/change_agent").glob("*/metrics.json"))) if (s/"baseline/change_agent").exists() else 0
    state=a.state or old.get("state") or ("running" if running else "not_started")
    if (s/"analysis/manifest.json").exists(): state="complete"
    payload={"updated_at":datetime.now().strftime("%F %T"),"state":state,"phase":a.phase or old.get("phase","pending"),"pid":pid,"running":running,"adapters":f"{adapters}/2","encoders":f"{enc}/2","captions":f"{captions}/5","model_metrics":f"{metrics}/155","changeagent_metrics":f"{baseline}/31"}
    target.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    pct=(adapters*5+enc*5+captions*5+metrics/155*50+baseline/31*5)
    html=f"""<!doctype html><html><head><meta charset='utf-8'><meta http-equiv='refresh' content='10'><title>Stage6.5</title><style>body{{font:16px system-ui;margin:32px;max-width:850px}}progress{{width:100%;height:28px}}pre{{background:#f4f4f4;padding:16px}}</style></head><body><h1>Stage6.5 calibrated-v3</h1><p><b>状态：</b>{state}　<b>阶段：</b>{payload['phase']}</p><progress max='100' value='{pct:.1f}'></progress><p>{pct:.1f}%</p><pre>{json.dumps(payload,ensure_ascii=False,indent=2)}</pre><p>每10秒自动刷新</p></body></html>"""
    (s/"status.html").write_text(html,encoding="utf-8")
    print(json.dumps(payload,ensure_ascii=False))


if __name__=="__main__": main()
