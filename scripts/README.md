# Public command wrappers

First materialize the validated stage layout, then invoke a concise public
entry point. Everything after `--` is passed to the original stage driver.

```bash
python scripts/train_partial_relkd.py --workspace /path/to/project -- \
  --config /path/to/project/configs/stage2/partial_relkd.json \
  --output /path/to/project/stage6_7_partial_hybrid/runs/p3_partial_relation
```

Run a wrapper with `--help` for its repository-level options and the target
stage driver directly for detailed model arguments.
