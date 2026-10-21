"""AS-IS → artifacts path mapping (pure data, no I/O)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CopySpec:
    """One file copy from AS-IS tree to fairai artifacts."""

    relative_source: str
    relative_dest: str
    role: str
    required: bool = True


# Canonical scientific artifacts (MVP).
CANONICAL_SPECS: tuple[CopySpec, ...] = (
    CopySpec(
        "DEVELOP/saved_models/van/v2/final_model.pt",
        "models/van/v2/final_model.pt",
        "van_v2_final",
    ),
    CopySpec(
        "DEVELOP/saved_models/van/v2/best_model.pt",
        "models/van/v2/best_model.pt",
        "van_v2_best",
    ),
    CopySpec(
        "DEVELOP/saved_models/van/v2/best_trial_params.json",
        "models/van/v2/best_trial_params.json",
        "van_v2_trial_params",
    ),
    CopySpec(
        "DEVELOP/saved_models/van/v2/best_threshold_params.json",
        "models/van/v2/best_threshold_params.json",
        "van_v2_threshold_params",
    ),
    CopySpec(
        "DEVELOP/saved_models/van/v1/final_model.pt",
        "models/van/v1/final_model.pt",
        "van_v1_final",
        required=False,
    ),
    CopySpec(
        "DEVELOP/saved_models/van/v1/best_model.pt",
        "models/van/v1/best_model.pt",
        "van_v1_best",
        required=False,
    ),
    CopySpec(
        "DEVELOP/saved_models/van/v1/best_trial_params.json",
        "models/van/v1/best_trial_params.json",
        "van_v1_trial_params",
        required=False,
    ),
    CopySpec(
        "DEVELOP/saved_models/van/v1/best_threshold_params.json",
        "models/van/v1/best_threshold_params.json",
        "van_v1_threshold_params",
        required=False,
    ),
    CopySpec(
        "DEVELOP/data/cross_related_dataset_van.jsonl",
        "datasets/processed/van_train_leakage_safe.jsonl",
        "van_train_leakage_safe",
    ),
    CopySpec(
        "DEVELOP/datasets/cross_related_dataset_van_final.jsonl",
        "datasets/processed/van_final.jsonl",
        "van_final",
    ),
    CopySpec(
        "DEVELOP/datasets/cross_related_dataset_van_final_minimal_dataset.jsonl",
        "datasets/processed/van_minimal_24k.jsonl",
        "van_minimal_24k",
    ),
    CopySpec(
        "DEVELOP/gold_scores.json",
        "results/gold_scores.json",
        "gold_scores",
    ),
    CopySpec(
        "DEVELOP/data/intersentence_train.json",
        "datasets/splits/stereoset/intersentence_train.json",
        "stereoset_inter_train",
    ),
    CopySpec(
        "DEVELOP/data/intersentence_test.json",
        "datasets/splits/stereoset/intersentence_test.json",
        "stereoset_inter_test",
    ),
    CopySpec(
        "DEVELOP/data/intrasentence_train.json",
        "datasets/splits/stereoset/intrasentence_train.json",
        "stereoset_intra_train",
    ),
    CopySpec(
        "DEVELOP/data/intrasentence_test.json",
        "datasets/splits/stereoset/intrasentence_test.json",
        "stereoset_intra_test",
    ),
    CopySpec(
        "DEVELOP/datasets/stereoset/stereoset_select_options.json",
        "datasets/processed/stereoset/stereoset_select_options.json",
        "stereoset_select_options",
        required=False,
    ),
    CopySpec(
        "DEVELOP/datasets/stereoset/stereoset_select_options_pred.json",
        "datasets/processed/stereoset/stereoset_select_options_pred.json",
        "stereoset_select_options_pred",
        required=False,
    ),
    CopySpec(
        "DEVELOP/datasets/stereoset/stereoset_select_options_pred_reduced.json",
        "datasets/processed/stereoset/stereoset_select_options_pred_reduced.json",
        "stereoset_select_options_pred_reduced",
        required=False,
    ),
    CopySpec(
        "DATASETS/CROWSPAIRS/crows_pairs_anonymized.csv",
        "datasets/raw/crows_pairs/crows_pairs_anonymized.csv",
        "crows_pairs_csv",
        required=False,
    ),
)
