# Experiments

`legacy_stages/` contains the exact stage-oriented drivers used for the
reported experiments. The stage names are retained for checkpoint lineage and
result traceability; new users normally enter through the top-level `scripts/`
wrappers and public `configs/`.

During materialization, every stage directory is copied back to the project
root so previously validated server commands and configuration paths continue
to work unchanged.
