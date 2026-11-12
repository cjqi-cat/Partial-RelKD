import argparse
from pathlib import Path
from common import load_json, allowed_degradations


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config", action="append", required=True); args=ap.parse_args()
    for path in args.config:
        c=load_json(path); missing=[]
        for key in ("dataset_root","metadata_root","change_agent_root","degradation_script","project_root"):
            if not Path(c[key]).exists(): missing.append(f"{key}={c[key]}")
        expected=allowed_degradations(c["family"])
        if c["allowed_degradations"] != expected: raise SystemExit(f"{path}: allowed mismatch {c['allowed_degradations']} != {expected}")
        if set(c["allowed_degradations"]) & set(c["heldout_degradations"]): raise SystemExit(f"{path}: LOFO leakage")
        vendor = {"chg2cap": "chg2cap", "mccformer_d": "sen", "rsiccformer": "sen",
                  "sen": "sen", "rscama": "rscama"}[c["model"]]
        vendor_root = Path(c["project_root"]) / "vendor" / vendor
        if not vendor_root.is_dir(): missing.append(f"vendor={vendor_root}")
        if missing: raise SystemExit(f"{path}: missing " + ", ".join(missing))
        print(f"OK {path}: {c['model']} allowed={expected}")
    print("PREFLIGHT_OK")
if __name__=="__main__": main()
