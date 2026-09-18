# Research artifact versioning policy

`outputs/` is working space for experiments and remains ignored by Git. The
exception is `outputs/frozen/`, a curated, version-controlled archive for
completed and frozen scientific experiments. Copy artifacts into this archive;
do not move them out of their original experiment output directories.

For a frozen experiment, normally preserve `best.pt` with Git LFS. Preserve
`latest.pt` only when it is scientifically necessary. Keep scientifically
relevant configs, environment metadata, histories, summaries, logs, per-case
metrics, aggregate metrics, evaluation provenance, statistical-analysis results,
paper tables and data, selected figures, and SHA256 manifests. Record SHA256
hashes for frozen checkpoints and important raw result artifacts. Every paper
table and figure must be traceable to the frozen raw files used to produce it.

Never commit raw BraTS data or NIfTI files. Never commit machine-specific paths,
including the server dataset root. Keep debug, smoke, benchmark, temporary,
cache, and disposable outputs outside the archive by default. The Git LFS rules
apply only to checkpoints under `outputs/frozen/`; other output directories
remain ignored.

Never run destructive cleanup such as `git clean -fdx` on a machine containing
unarchived research outputs.
