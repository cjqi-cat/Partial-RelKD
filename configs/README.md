# Public configuration entry points

- `stage1/geometry_quality.json`: change-sensitive partial visual adaptation.
- `stage2/partial_relkd.json`: main Partial-RelKD caption training.
- `stage2/partial_relkd_mask_free.json`: independently retrained mask-free
  lineage.
- `protocols/lofo_families.json`: canonical family membership and reporting
  rules.

The JSON files are path-tokenized templates. `tools/audit_release.py` checks
that the two main templates remain identical to their validated stage copies.
Use `tools/materialize_workspace.py` before execution.
