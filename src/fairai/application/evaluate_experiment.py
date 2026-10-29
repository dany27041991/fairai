"""Unified F1 + optional iCAT report for a train run directory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fairai.infrastructure.torch_van.checkpointing import write_json


@dataclass(slots=True)
class ExperimentEvalResult:
    run_dir: str
    kind: str
    classification: dict[str, Any] | None
    icat: dict[str, Any] | None
    out_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_dir": self.run_dir,
            "kind": self.kind,
            "classification": self.classification,
            "icat": self.icat,
            "out_path": self.out_path,
        }


def _detect_kind(run_dir: Path) -> str:
    if (run_dir / "logistic.joblib").is_file():
        return "frozen_lr"
    if (run_dir / "best_model.pt").is_file() or (run_dir / "final_model.pt").is_file():
        return "van"
    raise FileNotFoundError(f"No checkpoint in {run_dir}")


def evaluate_experiment_run(
    *,
    run_dir: Path,
    eval_dataset: Path | None = None,
    splits_dir: Path | None = None,
    device: str = "cpu",
    batch_size: int = 32,
    max_samples: int | None = None,
    subset: str = "all",
    out_name: str = "eval_report.json",
    eval_out_dir: Path | None = None,
) -> ExperimentEvalResult:
    """Load ``config.snapshot.json`` + checkpoint; write F1 and/or iCAT."""
    run_dir = Path(run_dir)
    cfg_path = run_dir / "config.snapshot.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Missing {cfg_path}")

    kind = _detect_kind(run_dir)
    classification: dict[str, Any] | None = None
    icat: dict[str, Any] | None = None

    if kind == "frozen_lr":
        model_path = run_dir / "logistic.joblib"
        if eval_dataset is not None:
            from fairai.application.evaluate_frozen_lr import evaluate_frozen_lr

            classification = evaluate_frozen_lr(
                model_path=model_path,
                dataset_path=eval_dataset,
                device=device,
                batch_size=batch_size,
                max_samples=max_samples,
            ).to_dict()
        if splits_dir is not None:
            from fairai.application.evaluate_icat_frozen_lr import evaluate_icat_frozen_lr

            icat = evaluate_icat_frozen_lr(
                model_path=model_path,
                splits_dir=splits_dir,
                subset=subset,
                device=device,
                batch_size=batch_size,
            ).to_dict()
    else:
        ckpt = run_dir / "best_model.pt"
        if not ckpt.is_file():
            ckpt = run_dir / "final_model.pt"
        if eval_dataset is not None:
            from fairai.application.evaluate_classification import evaluate_classification

            classification = evaluate_classification(
                checkpoint_path=ckpt,
                config_path=cfg_path,
                dataset_path=eval_dataset,
                device=device,
                batch_size=batch_size,
                max_samples=max_samples,
            ).to_dict()
        if splits_dir is not None:
            from fairai.application.evaluate_icat import evaluate_icat

            icat = evaluate_icat(
                checkpoint_path=ckpt,
                config_path=cfg_path,
                splits_dir=splits_dir,
                subset=subset,
                device=device,
                batch_size=batch_size,
            ).to_dict()

    payload = {
        "run_dir": str(run_dir),
        "kind": kind,
        "classification": classification,
        "icat": icat,
        "summary": {
            "f1_macro": (classification or {}).get("f1_macro"),
            "accuracy": (classification or {}).get("accuracy"),
            "avg_icat": (icat or {}).get("avg_icat"),
            "intersentence_icat": ((icat or {}).get("intersentence") or {}).get("icat"),
            "intrasentence_icat": ((icat or {}).get("intrasentence") or {}).get("icat"),
        },
    }
    if eval_out_dir is not None:
        out_path = Path(eval_out_dir) / f"{run_dir.name}_{out_name}"
    else:
        out_path = run_dir / out_name
    try:
        write_json(out_path, payload)
    except OSError:
        fallback = Path("artifacts/results/metrics/eval_reports") / f"{run_dir.name}_{out_name}"
        write_json(fallback, payload)
        out_path = fallback
    return ExperimentEvalResult(
        run_dir=str(run_dir),
        kind=kind,
        classification=classification,
        icat=icat,
        out_path=str(out_path),
    )


def compare_experiment_runs(
    *,
    run_dirs: list[Path],
    eval_dataset: Path | None = None,
    splits_dir: Path | None = None,
    device: str = "cpu",
    batch_size: int = 32,
    max_samples: int | None = None,
    subset: str = "all",
    out_path: Path,
    eval_out_dir: Path | None = None,
) -> dict[str, Any]:
    """Evaluate many runs and write a paper-ready comparison table."""
    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        result = evaluate_experiment_run(
            run_dir=run_dir,
            eval_dataset=eval_dataset,
            splits_dir=splits_dir,
            device=device,
            batch_size=batch_size,
            max_samples=max_samples,
            subset=subset,
            eval_out_dir=eval_out_dir,
        )
        metrics_path = Path(run_dir) / "metrics.json"
        train_meta: dict[str, Any] = {}
        if metrics_path.is_file():
            import json

            train_meta = json.loads(metrics_path.read_text(encoding="utf-8"))
        summary = {
            "run_dir": result.run_dir,
            "kind": result.kind,
            "experiment": train_meta.get("experiment"),
            "ablation": train_meta.get("ablation"),
            "best_val_f1_macro": train_meta.get("best_val_f1_macro"),
            "eval_f1_macro": (result.classification or {}).get("f1_macro"),
            "eval_accuracy": (result.classification or {}).get("accuracy"),
            "avg_icat": (result.icat or {}).get("avg_icat"),
            "icat_subset": (result.icat or {}).get("subset"),
            "by_bias_type_f1": (result.classification or {}).get("by_bias_type"),
            "by_bias_type_icat_inter": (result.icat or {}).get(
                "by_bias_type_intersentence"
            ),
            "by_bias_type_icat_intra": (result.icat or {}).get(
                "by_bias_type_intrasentence"
            ),
        }
        rows.append(summary)

    table = {"n": len(rows), "rows": rows}
    write_json(out_path, table)
    return table
