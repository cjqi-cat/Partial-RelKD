from pathlib import Path
import importlib.util

def _module(cfg):
    path=Path(cfg["stage10_root"])/"runner/models.py"
    spec=importlib.util.spec_from_file_location("authoritative_models",path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module

def build_model(cfg,vocab):
    base=dict(cfg); base["project_root"]=cfg["stage10_root"]
    base.setdefault("output_root",str(Path(cfg["stage10_root"])/"runs/stage109_adapter_logs"/cfg["model"]))
    Path(base["output_root"]).mkdir(parents=True,exist_ok=True)
    return _module(cfg).build_model(base,vocab)
