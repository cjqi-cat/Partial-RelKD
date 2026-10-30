# Git release checklist

## Safe to commit now

- [x] Core P1/P2/P3 training and evaluation code
- [x] Calibrated-v3 degradation operators
- [x] Mask-free extension and ablations
- [x] Severity, multi-seed, full-family, and Bootstrap scripts
- [x] Three-dataset target-supervised reproduction scripts
- [x] Compact aggregate result summaries
- [x] Method overview figure
- [x] `.gitignore` for data, weights, caches, predictions, and logs
- [x] Machine-specific paths replaced by materialization tokens

## Required before making the repository public

- [x] License decision recorded: no explicit repository license is granted
- [x] Record exact upstream URLs, commits, and detected license status
- [ ] Decide where approved checkpoints will be hosted (release assets, model
      hub, or institutional storage)
- [x] Run the data-free core smoke test from a fresh clone
- [x] Verify that no dataset files, model weights, or third-party source trees are staged
- [x] Verify that reported aggregate files match the submitted manuscript

## May be deferred during double-blind review

- [ ] Add final author names and `CITATION.cff`
- [ ] Add the paper/preprint URL
- [ ] Publish approved checkpoints and insert their stable download URLs
- [ ] Archive a versioned release with a DOI (for example, through Zenodo)

Do not add author-identifying metadata to an anonymous review repository unless
the venue explicitly permits it.

## Recommended Git sequence

```bash
cd Partial-RelKD-GitHub
python tools/audit_release.py
python tools/smoke_test.py
git init
git add .
git status --short
git diff --cached --stat
git commit -m "Initial Partial-RelKD research-code release"
```

Inspect `git status` and the staged diff before adding a remote. Do not use
`git add` from the parent research workspace.
