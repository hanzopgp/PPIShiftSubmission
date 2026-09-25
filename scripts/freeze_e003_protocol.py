"""Create the immutable Core-4 E003 protocol freeze from input artifacts.

The freeze is deliberately generated only from corrected panels, configuration,
and request/runtime manifests. It does not inspect predictions or any test
metric, so it can be committed before corrected test inference begins. The
historical SENSE-PPI cache helpers below remain available for audit tests and
legacy artifacts, but SENSE-PPI is not a Core-4 freeze prerequisite.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ppi_shift.artifacts import git_commit, sha256_path, write_yaml
from ppi_shift.e003_method_fidelity import E003_ENVIRONMENTS

UTC = getattr(timezone, "utc")

EXPECTED_BRANCH = "historical-e003-freeze"
CORE_MODEL_ROSTER = (
    "symmetric_frozen_esm2_pair_mlp",
    "dscript_esm2",
    "tuna_corrected",
    "rapppid",
)

SENSE_CACHE_MODEL = "facebook/esm2_t36_3B_UR50D"
SENSE_CACHE_REVISION = "476b639933c8baad5ad09a60ac1a87f987b656fc"
SENSE_CACHE_SHARD_COUNT = 16
SENSE_CACHE_MAXIMUM_LENGTH = 512
SENSE_CACHE_EMBEDDING_DIMENSION = 2560


def _hash_tree(root: Path) -> dict[str, Any]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    records: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        records.append(
            {"path": relative, "sha256": sha256_path(path), "bytes": path.stat().st_size}
        )
    digest_input = "\n".join(f"{row['path']}\t{row['sha256']}" for row in records) + "\n"
    import hashlib

    return {
        "path": str(root),
        "files": len(records),
        "bytes": sum(int(row["bytes"]) for row in records),
        "tree_sha256": hashlib.sha256(digest_input.encode("utf-8")).hexdigest(),
        "summaries": [
            {"path": row["path"], "sha256": row["sha256"], "bytes": row["bytes"]}
            for row in records
            if row["path"].startswith("summary-")
        ],
    }


def _load_mapping(path: Path) -> dict[str, Any]:
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping YAML: {path}")
    return value


def _git_branch(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "branch", "--show-current"],
        check=False,
        capture_output=True,
        text=True,
    )
    branch = result.stdout.strip()
    if result.returncode != 0 or not branch:
        raise SystemExit(
            f"Unable to resolve the current Git branch for protocol freeze: {repo_root}"
        )
    return branch


def _validate_core_model_config(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    models = config.get("models", {})
    negative_environments = config.get("negative_environments", {})
    if not isinstance(models, dict) or not isinstance(negative_environments, dict):
        raise SystemExit("E003 config lacks model or negative-environment mappings")
    required = tuple(models.get("required", ()))
    if required != CORE_MODEL_ROSTER:
        raise SystemExit(
            "E003 Core-4 model roster mismatch: "
            f"observed={list(required)} expected={list(CORE_MODEL_ROSTER)}"
        )
    excluded = models.get("excluded_from_core", {})
    senseppi = excluded.get("senseppi") if isinstance(excluded, dict) else None
    if not isinstance(senseppi, dict) or senseppi.get("status") != "optional_future_supplementary":
        raise SystemExit(
            "E003 config must explicitly record SENSE-PPI as optional_future_supplementary"
        )
    source_revisions = models.get("source_revisions", {})
    model_hyperparameters = models.get("fidelity", {})
    if not isinstance(source_revisions, dict) or not isinstance(model_hyperparameters, dict):
        raise SystemExit("E003 config lacks source revisions or model hyperparameters")
    matrix = config.get("matrix", {})
    if not isinstance(matrix, dict):
        raise SystemExit("E003 config lacks a matrix mapping")
    expected_matrix = {
        "models": 4,
        "train_environments": 6,
        "test_environments": 6,
        "total_core_cells": 432,
        "training_units": 72,
    }
    mismatches = {
        key: {"expected": value, "observed": matrix.get(key)}
        for key, value in expected_matrix.items()
        if matrix.get(key) != value
    }
    if mismatches or tuple(matrix.get("seeds", ())) != (11, 22, 33):
        if tuple(matrix.get("seeds", ())) != (11, 22, 33):
            mismatches["seeds"] = {"expected": [11, 22, 33], "observed": matrix.get("seeds")}
        raise SystemExit(f"E003 Core-4 matrix mismatch: {mismatches}")
    declared_core = tuple(negative_environments.get("core", ()))
    if declared_core != E003_ENVIRONMENTS:
        raise SystemExit(
            "E003 core negative-environment roster mismatch: "
            f"observed={list(declared_core)} expected={list(E003_ENVIRONMENTS)}"
        )
    return models, negative_environments


def _required_artifact_specs(paths: list[Path]) -> list[dict[str, str]]:
    required_names = {
        "e003_backend_request_manifest.json",
        "e003_python_request_manifest.json",
        "e003_frozen_request_manifest.json",
        "e003_reuse_request_manifest.json",
    }
    if not paths:
        raise SystemExit(
            "At least one --required-artifact is required so the freeze binds the prepared "
            "Core-4 request/runtime inputs"
        )
    observed_names = {path.name for path in paths}
    missing_names = sorted(required_names - observed_names)
    if missing_names:
        raise SystemExit(
            "Core-4 freeze is missing required request manifests: "
            f"{missing_names}"
        )
    specs: list[dict[str, str]] = []
    seen: set[str] = set()
    for path in paths:
        resolved = str(path)
        if resolved in seen:
            raise SystemExit(f"Duplicate required artifact: {path}")
        seen.add(resolved)
        if not path.is_file():
            raise SystemExit(f"Required artifact is missing: {path}")
        specs.append({"path": resolved, "sha256": sha256_path(path)})
    return specs


def _validate_sense_cache_summaries(
    cache_root: Path,
) -> list[Path]:
    summaries = sorted(cache_root.glob("summary-*.json"))
    expected_names = {
        f"summary-shard-{index:03d}-of-{SENSE_CACHE_SHARD_COUNT:03d}.json"
        for index in range(SENSE_CACHE_SHARD_COUNT)
    }
    observed_names = {path.name for path in summaries}
    if observed_names != expected_names:
        missing = sorted(expected_names - observed_names)
        unexpected = sorted(observed_names - expected_names)
        raise SystemExit(
            "SENSE cache shard summary names are invalid: "
            f"missing={missing}, unexpected={unexpected}"
        )
    values: list[dict[str, Any]] = []
    for path in summaries:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise SystemExit(f"SENSE cache summary is not a JSON object: {path}")
        values.append(value)
    registry_paths = {str(item.get("registry_path", "")) for item in values}
    registry_hashes = {str(item.get("registry_sha256", "")) for item in values}
    for index, item in enumerate(values):
        expected = {
            "model_name": SENSE_CACHE_MODEL,
            "model_revision": SENSE_CACHE_REVISION,
            "maximum_sequence_length": SENSE_CACHE_MAXIMUM_LENGTH,
            "shard_index": index,
            "shard_count": SENSE_CACHE_SHARD_COUNT,
            "dtype": "float16",
            "special_tokens_excluded": True,
            "embedding_dimensions": [SENSE_CACHE_EMBEDDING_DIMENSION],
        }
        mismatches = {
            key: {"expected": expected_value, "observed": item.get(key)}
            for key, expected_value in expected.items()
            if item.get(key) != expected_value
        }
        if mismatches:
            raise SystemExit(
                f"SENSE cache shard {index} violates the frozen representation contract: "
                f"{mismatches}"
            )
        if int(item.get("assigned_records", 0)) < 1:
            raise SystemExit(f"SENSE cache shard {index} is empty")
    if len(registry_paths) != 1 or not next(iter(registry_paths), ""):
        raise SystemExit("SENSE cache shards do not share one non-empty registry path")
    if len(registry_hashes) != 1 or not next(iter(registry_hashes), ""):
        raise SystemExit("SENSE cache shards do not share one non-empty registry SHA256")
    return summaries


def _validate_sense_cache_registry_manifest(
    summaries: list[Path], manifest_path: Path
) -> dict[str, str]:
    """Bind the derived hash-keyed registry manifest to every cache shard."""

    values = [json.loads(path.read_text(encoding="utf-8")) for path in summaries]
    registry_paths = {str(item.get("registry_path", "")) for item in values}
    registry_hashes = {str(item.get("registry_sha256", "")) for item in values}
    registry_manifest = _load_mapping(manifest_path)
    manifest_registry_path = str(registry_manifest.get("fasta_path", ""))
    manifest_registry_hash = str(registry_manifest.get("fasta_sha256", ""))
    if (
        registry_manifest.get("status") != "complete"
        or len(registry_paths) != 1
        or len(registry_hashes) != 1
        or manifest_registry_path != next(iter(registry_paths), "")
        or manifest_registry_hash != next(iter(registry_hashes), "")
    ):
        raise SystemExit(
            "SENSE cache registry manifest does not match the completed cache summaries"
        )
    registry_file = Path(manifest_registry_path)
    if not registry_file.is_file() or sha256_path(registry_file) != manifest_registry_hash:
        raise SystemExit(
            "SENSE cache registry FASTA is missing or its hash does not match the manifest"
        )
    return {
        "path": str(manifest_path),
        "sha256": sha256_path(manifest_path),
        "fasta_path": manifest_registry_path,
        "fasta_sha256": manifest_registry_hash,
    }


def _panel_hashes(panel: dict[str, Any]) -> dict[str, dict[str, str]]:
    stages = panel.get("stages", {})
    if not isinstance(stages, dict):
        raise SystemExit("E003 panel manifest lacks a stages mapping")
    result: dict[str, dict[str, str]] = {}
    for stage in ("train", "validation", "test"):
        if stage not in stages:
            raise SystemExit(f"E003 panel manifest is missing stage: {stage}")
        stage_value = stages[stage]
        if not isinstance(stage_value, dict):
            raise SystemExit(f"E003 panel stage is not a mapping: {stage}")
        samplers = stage_value.get("samplers", {})
        if not isinstance(samplers, dict):
            raise SystemExit(f"E003 panel stage lacks a samplers mapping: {stage}")
        result[stage] = {}
        for environment in E003_ENVIRONMENTS:
            spec = samplers.get(environment)
            if not isinstance(spec, dict) or not str(spec.get("panel_sha256", "")):
                raise SystemExit(
                    f"E003 panel sampler lacks a non-empty panel_sha256: {stage}/{environment}"
                )
            result[stage][str(environment)] = str(spec["panel_sha256"])
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--required-artifact",
        type=Path,
        action="append",
        default=[],
        help=(
            "Prepared Core-4 request/runtime manifest to bind by SHA256; repeat for "
            "each required input."
        ),
    )
    parser.add_argument("--output-yaml", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--external-manifest", type=Path)
    parser.add_argument("--datasail-manifest", type=Path)
    args = parser.parse_args()
    for path in (args.output_yaml, args.output_markdown):
        if path.exists():
            raise SystemExit(f"Refusing to overwrite existing protocol freeze: {path}")

    panel = _load_mapping(args.panel_manifest)
    config = _load_mapping(args.config)
    if config.get("experiment_id") != "E003":
        raise SystemExit("Unexpected E003 config experiment_id")
    models, negative_environments = _validate_core_model_config(config)
    if tuple(panel.get("environments", ())) != E003_ENVIRONMENTS:
        raise SystemExit(
            "E003 panel negative-environment roster mismatch: "
            f"observed={panel.get('environments')} expected={list(E003_ENVIRONMENTS)}"
        )
    source_revisions = models.get("source_revisions", {})
    model_hyperparameters = models.get("fidelity", {})
    required_artifacts = _required_artifact_specs(args.required_artifact)
    branch = _git_branch(args.repo_root)
    if branch != EXPECTED_BRANCH:
        raise SystemExit(
            f"Protocol freeze must be generated on {EXPECTED_BRANCH}, observed {branch}"
        )

    def file_spec(path: Path | None) -> dict[str, Any] | None:
        if path is None:
            return None
        return {"path": str(path), "sha256": sha256_path(path)}

    freeze = {
        "manifest_type": "ppi_shift.e003_protocol_freeze",
        "manifest_version": 1,
        "experiment_id": "E003",
        "freeze_status": "protocol_frozen_before_any_corrected_test_score",
        "freeze_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "branch": branch,
        "base_git_sha": git_commit(args.repo_root),
        "corrected_source_sha": git_commit(args.repo_root),
        "motivation": (
            "E003 corrects method-fidelity problems discovered during review of "
            "immutable E002 evidence."
        ),
        "config": {"path": str(args.config), "sha256": sha256_path(args.config)},
        "matrix": config.get("matrix", {}),
        "required_core_artifacts": required_artifacts,
        "source_revisions": source_revisions,
        "model_hyperparameters": model_hyperparameters,
        "sampler_protocol": {
            "bernett2026_ilp_exact": negative_environments.get(
                "bernett2026_ilp_exact", {}
            ),
            "tppni2025_scm_cl3_exact": negative_environments.get(
                "tppni2025_scm_cl3_exact", {}
            ),
            "tppni2025_scm_cl3_exact_balanced": negative_environments.get(
                "tppni2025_scm_cl3_exact_balanced", {}
            ),
            "tppni2025_scm_cl3_leakage_safe": negative_environments.get(
                "tppni2025_scm_cl3_leakage_safe", {}
            ),
        },
        "panel_hashes": _panel_hashes(panel),
        "corrected_panel_manifest": {
            "path": str(args.panel_manifest),
            "sha256": sha256_path(args.panel_manifest),
            "schema_version": panel.get("schema_version"),
            "artifact_root": str(args.panel_manifest.parent),
            "stages": panel.get("stages", {}),
        },
        "upstream": {
            "bernett": {
                "repository": "https://github.com/bionetslab/ppi-splitting-pipeline",
                "revision": "8bf9adb183b702c88db62ff878de482e2f20e4b2",
                "implementation": "pinned-source-adapted-global-pool-ilp",
            },
            "tppni": {
                "repository": "https://github.com/jbusto/UPNA-PPI",
                "revision": "20d1d8225e9d377434669973b7391697be6b4aa2",
                "equation": "p_ij=1/(exp(lambda_i+lambda_j)+1)",
            },
            "rapppid": {
                "repository": "https://github.com/jszym/rapppid",
                "revision": "c3a28be56fb3bd96ccf8cc44ea9145cfcadeaf6c",
            },
        },
        "annotations": panel.get("annotations", {}),
        "model_roster": list(CORE_MODEL_ROSTER),
        "excluded_models": {
            "senseppi": models["excluded_from_core"]["senseppi"],
        },
        "negative_environment_roster": panel.get("environments", []),
        "roles": {
            "core": [
                "uniform_partner",
                "degree_partner_matched",
                "sequence_nuisance_matched",
                "esm_partner_matched",
                "bernett2026_ilp_exact",
                "tppni2025_scm_cl3_exact_balanced",
            ],
            "published_method_supplementary": ["tppni2025_scm_cl3_exact"],
            "leakage_safe_supplementary": ["tppni2025_scm_cl3_leakage_safe"],
            "measured_transfer": ["vidal", "intact"],
        },
        "seeds": [11, 22, 33],
        "sampler_seed": 1729,
        "selection": {
            "checkpoint": "maximum matched-validation AUPRC",
            "threshold": "matched-validation maximum MCC",
            "test_adaptation": False,
            "external_selection": False,
        },
        "metrics": ["auprc", "auroc", "mcc", "balanced_accuracy", "f1", "brier", "nll", "ece"],
        "uncertainty": {
            "bootstrap_replicates": 2000,
            "bootstrap_seed": 1729,
            "global_pool_scheme": (
                "stratified positives and negatives; shared resamples within comparison"
            ),
            "cluster_sensitivity": (
                "protein-endpoint and positive-backbone sensitivity; limitations reported"
            ),
            "seed_stability": "all 3x3 training-seed pairings, separate estimand",
        },
        "external_protocol": {
            "no_external_retraining": True,
            "no_external_threshold_selection": True,
            "vidal": "frozen measured-assay scoring-only cohort",
            "intact": "full prevalence-normalized analysis plus post-hoc balanced sensitivity",
            "biogrid_string": "source-specific synthetic-unobserved-pair surfaces",
        },
        "datasail": file_spec(args.datasail_manifest),
        "external_manifest": file_spec(args.external_manifest),
        "figure_endpoints": {
            "directory": "paper/figures/e003",
            "formats": ["pdf", "svg", "png"],
            "source_data_adjacent": True,
        },
    }
    write_yaml(args.output_yaml, freeze)
    lines = [
        "# E003 protocol freeze",
        "",
        "> Frozen before any corrected E003 test inference or result inspection.",
        "",
        (
            "E003 was motivated by method-fidelity problems discovered during review of "
            "immutable E002 evidence. The corrected sampler definitions, model roster, "
            "validation-owned selection rules, and uncertainty estimands below were "
            "fixed before observing E003 test outcomes."
        ),
        "",
        f"- Branch: `{freeze['branch']}`",
        f"- Base Git SHA: `{freeze['base_git_sha']}`",
        f"- Corrected source SHA: `{freeze['corrected_source_sha']}`",
        f"- Panel manifest: `{args.panel_manifest}` ({sha256_path(args.panel_manifest)})",
        f"- Config: `{args.config}` ({sha256_path(args.config)})",
        "- Core roster: `symmetric_frozen_esm2_pair_mlp`, `dscript_esm2`, "
        "`tuna_corrected`, `rapppid` (432 cells; 72 training units).",
        "- SENSE-PPI is explicitly excluded from Core-4 and retained only as a "
        "historical/optional supplementary model; no SENSE cache or job is required.",
        "- Required Core-4 request/runtime manifests:",
        *[
            f"  - `{spec['path']}` ({spec['sha256']})"
            for spec in required_artifacts
        ],
        "- Model source revisions, model hyperparameters, sampler definitions, "
        "TPPNI topology provenance, and train/validation/test panel hashes are "
        "copied into the machine-readable freeze.",
        "",
        "## Frozen scientific boundaries",
        "",
        (
            "- All six core panels are balanced and retain `synthetic_unobserved_pair` "
            "semantics; no synthetic complement pair is called a biological "
            "non-interaction."
        ),
        (
            "- Bernett is the new global-pool `bernett2026_ilp_exact` implementation; "
            "the historical E002 anchor-fixed adaptation is not reused or renamed "
            "in artifacts."
        ),
        "- Paper-faithful TPPNI and leakage-safe TPPNI remain separate supplementary surfaces.",
        (
            "- Checkpoint selection uses only matched validation AUPRC; thresholds "
            "use only matched validation maximum MCC."
        ),
        (
            "- External labels cannot affect training, checkpoint selection, threshold "
            "selection, or model naming."
        ),
        (
            "- Primary ranking estimates average the three training seeds; evaluation "
            "bootstrap and seed stability are reported separately."
        ),
        "",
        "## Artifact manifest",
        "",
        "The machine-readable freeze is authoritative: `data/manifests/e003_protocol_freeze.yaml`.",
        (
            "Panel, annotation, sampler-pool, Core-4 request/runtime, external, "
            "DataSAIL, model, seed, metric, uncertainty, and figure contracts are "
            "recorded there with hashes where available."
        ),
        "",
    ]
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "yaml": str(args.output_yaml),
                "markdown": str(args.output_markdown),
                "git_sha": freeze["base_git_sha"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
