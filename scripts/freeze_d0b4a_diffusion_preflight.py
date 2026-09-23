"""Freeze the D0-B4a engineering preflight using only standard-library CPU IO."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

BRANCH = "experiment/d0b4-diffusion-design"
DEFAULT_INPUT = Path("outputs/d0b4_diffusion_preflight")
DEFAULT_OUTPUT = Path("outputs/frozen/d0b4a_diffusion_memory_preflight")
MICROBATCHES = (8, 4, 2, 1)
GPU = "NVIDIA GeForce RTX 3090"
RULE = "largest passing physical microbatch with peak_reserved_fraction <= 0.90"
FLAGS = ("pass", "loss_finite", "gradient_finite", "all_za_shapes_valid",
         "all_output_shapes_valid", "source_gradient_free", "optimizer_step_performed",
         "optimizer_state_materialized", "memory_headroom_pass")
NON_SCIENTIFIC = dict(scientific_training=False, validation_used=False,
                      model_selection=False, scientific_interpretation=False)
FINGERPRINTS = dict(
    effective_batch_fingerprint="17743ab4071800a24d9a572519bee6577deda903b0078a8a2aa71845fc24ab4a",
    randomness_fingerprint="02450e90fee61a3907004493442320b261f7aeabfabd21e6614d4c468671c003")
SELECTED_MEMORY = dict(peak_memory_allocated_bytes=3961428992,
                       peak_memory_reserved_bytes=4150263808,
                       peak_allocated_fraction=0.15660270779844918,
                       peak_reserved_fraction=0.16406770176197644)
PROVENANCE_KEYS = ("gpu_name", "torch_version", "cuda_version", "seed",
                   "patch_manifest_sha256", "normalization_contract_sha256",
                   "source_checkpoint_sha256", "d0b3_decision_sha256")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def encode(payload):
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def read_payload(data):
    payload = json.loads(data)
    if not isinstance(payload, dict):
        raise ValueError("Expected JSON object")
    return payload


def validate_candidates(candidates):
    if set(candidates) != set(MICROBATCHES):
        raise ValueError("Required microbatches are exactly {8,4,2,1}")
    for mb, candidate in candidates.items():
        def require(condition, message):
            if not condition:
                raise ValueError(f"microbatch {mb}: {message}")
        for key, expected in (("microbatch", mb), ("effective_batch_size", 8),
                              ("gradient_accumulation_steps", 8 // mb),
                              ("model_parameter_count", 22825280)):
            require(type(candidate.get(key)) is int and candidate[key] == expected, key)
        for key in FLAGS:
            require(candidate.get(key) is True, key)
        require(candidate.get("oom") is False, "oom")
        require("failure_reason" in candidate and candidate["failure_reason"] is None, "failure_reason")
        require(candidate.get("gpu_name") == GPU, "gpu_name")
        for key in NON_SCIENTIFIC:
            require(candidate.get(key) is False, key)
        for key, expected in FINGERPRINTS.items():
            require(candidate.get(key) == expected, f"{key} mismatch")
        for key in SELECTED_MEMORY:
            value = candidate.get(key)
            require(type(value) in (int, float) and math.isfinite(value) and value >= 0, key)
            if key.endswith("_bytes"):
                require(type(value) is int, key)
            else:
                require(value <= 1, key)
        require(candidate["peak_reserved_fraction"] <= 0.90, "peak_reserved_fraction > 0.90")
    # No elapsed-time field is consulted by this pre-specified rule.
    selected = max(mb for mb, c in candidates.items()
                   if c["pass"] is True and c["peak_reserved_fraction"] <= 0.90)
    if selected != 8 or any(candidates[selected][k] != v for k, v in SELECTED_MEMORY.items()):
        raise ValueError("Selected candidate does not match the locked microbatch-8 memory result")
    return selected


def decision_record(candidates):
    selected = validate_candidates(candidates)
    return dict(stage="D0-B4a", purpose="GPU memory and shape preflight", **NON_SCIENTIFIC,
                selection_rule=RULE, selected_physical_microbatch=selected,
                gradient_accumulation_steps=1, effective_batch_size=8,
                **{f"selected_{k}": candidates[selected][k] for k in SELECTED_MEMORY},
                **FINGERPRINTS,
                candidate_memory_summaries=[dict(physical_microbatch=mb,
                    gradient_accumulation_steps=8 // mb, effective_batch_size=8,
                    **{k: candidates[mb][k] for k in SELECTED_MEMORY}) for mb in MICROBATCHES])


def summary_record(decision):
    return dict(stage="D0-B4a", **NON_SCIENTIFIC, all_four_candidates_passed=True,
                selection_based_on_elapsed_time=False, selection_rule=RULE,
                selected_physical_microbatch=8, gradient_accumulation_steps=1,
                effective_batch_size=8,
                candidate_memory_summaries=decision["candidate_memory_summaries"],
                statement="All four candidates passed. Selection was NOT based on elapsed time. "
                "No scientific diffusion training occurred; no validation data were used. "
                "Physical microbatch 8 is locked for the current D0-B4 training contract. "
                "If training later uses physical microbatch <8 with accumulation, "
                "loss-normalization equivalence must be explicitly re-audited.")


def candidate_provenance(candidates):
    return {str(mb): {k: candidates[mb][k] for k in PROVENANCE_KEYS if k in candidates[mb]}
            for mb in MICROBATCHES}


def runtime_git_provenance():
    root = Path(__file__).resolve().parents[1]
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root, text=True).strip()
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    if branch != BRANCH:
        raise ValueError(f"Expected branch {BRANCH}; got {branch}")
    return dict(runtime_git_branch=branch, runtime_git_commit=git("rev-parse", "HEAD"),
                git_status_short=git("status", "--short").splitlines())


def verify(output_dir):
    root = Path(output_dir)
    if root.is_symlink() or (root / "MANIFEST.sha256").is_symlink():
        raise ValueError("Symlink in frozen artifact")
    files = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Symlink in frozen artifact: {path}")
        if path.is_file() and path != root / "MANIFEST.sha256":
            files[path.relative_to(root).as_posix()] = path.read_bytes()
    expected_manifest = "".join(f"{sha256(files[name])}  {name}\n" for name in sorted(files))
    if (root / "MANIFEST.sha256").read_text() != expected_manifest:
        raise ValueError("MANIFEST mismatch: missing, extra, changed, or noncanonical entry")
    candidates = {mb: read_payload(files[f"candidates/microbatch_{mb}.json"]) for mb in MICROBATCHES}
    expected = decision_record(candidates)
    if read_payload(files["decision/microbatch_decision.json"]) != expected:
        raise ValueError("Decision consistency mismatch")
    if read_payload(files["d0b4a_summary.json"]) != summary_record(expected):
        raise ValueError("Summary consistency mismatch")
    provenance = read_payload(files["provenance.json"])
    if provenance.get("candidate_provenance") != candidate_provenance(candidates):
        raise ValueError("Candidate provenance mismatch")
    for key, value in dict(selected_physical_microbatch=8, gradient_accumulation_steps=1,
                           effective_batch_size=8, gpu_name=GPU).items():
        if provenance.get(key) != value:
            raise ValueError(f"Provenance mismatch: {key}")
    records = provenance["source_files"]
    copied = {name for name in files if name.startswith(("candidates/", "failed_attempts/"))}
    if len(records) != len(copied) or {r["frozen_path"] for r in records} != copied:
        raise ValueError("Source file inventory mismatch")
    for record in records:
        digest = sha256(files[record["frozen_path"]])
        if record["source_sha256"] != digest or record["frozen_copy_sha256"] != digest:
            raise ValueError("Source/frozen copy SHA256 mismatch")
    return expected


def run(input_dir=DEFAULT_INPUT, output_dir=DEFAULT_OUTPUT, *, verify_only=False):
    source, destination = Path(input_dir).expanduser(), Path(output_dir).expanduser()
    if verify_only:
        return verify(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Refusing to overwrite frozen directory: {destination}")
    files, records, candidates = {}, [], {}
    def preserve(path, relative):
        data = path.read_bytes()
        files[relative] = data
        records.append(dict(source_path=str(path.resolve()), source_sha256=sha256(data),
                            frozen_path=relative, frozen_copy_sha256=sha256(data)))
        return read_payload(data)
    for mb in MICROBATCHES:
        candidates[mb] = preserve(source / f"microbatch_{mb}_retry1.json", f"candidates/microbatch_{mb}.json")
    decision = decision_record(candidates)
    required = {f"microbatch_{mb}_retry1.json" for mb in MICROBATCHES}
    for path in sorted(source.glob("*.json")):
        if path.name in required:
            continue
        payload = read_payload(path.read_bytes())
        if payload.get("pass") is False or payload.get("oom") is True or payload.get("failure_reason") is not None:
            preserve(path, f"failed_attempts/{path.name}")
    provenance = dict(**runtime_git_provenance(), source_files=records,
                      candidate_provenance=candidate_provenance(candidates), gpu_name=GPU,
                      selected_physical_microbatch=8, gradient_accumulation_steps=1, effective_batch_size=8)
    files["decision/microbatch_decision.json"] = encode(decision)
    files["d0b4a_summary.json"] = encode(summary_record(decision))
    files["provenance.json"] = encode(provenance)
    files["MANIFEST.sha256"] = "".join(f"{sha256(files[name])}  {name}\n" for name in sorted(files)).encode()
    # Exclusive creation, after all inputs and serialization have been validated.
    destination.mkdir(parents=True, exist_ok=False)
    for name, data in files.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(data)
    return verify(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    decision = run(args.input_dir, args.output_dir, verify_only=args.verify_only)
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
