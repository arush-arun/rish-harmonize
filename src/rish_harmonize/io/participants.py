"""
Participants I/O Module

Parse BIDS participants.tsv or CSV files with subject demographics
for covariate-adjusted RISH harmonization.
"""

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ParticipantData:
    """Parsed participant demographics.

    Attributes
    ----------
    subject_ids : list of str
        Subject identifiers (ordered to match image list)
    covariates : dict
        Covariate name -> list of float values (same order as subject_ids)
    """
    subject_ids: List[str] = field(default_factory=list)
    covariates: Dict[str, List[float]] = field(default_factory=dict)

    @property
    def n_subjects(self) -> int:
        return len(self.subject_ids)

    @property
    def covariate_names(self) -> List[str]:
        return list(self.covariates.keys())


def _encode_categorical(values: List[str]) -> List[float]:
    """Encode categorical values as numeric.

    Currently handles sex/gender: M/Male/1 -> 1.0, F/Female/0 -> 0.0.
    For other categorical variables, encodes unique levels as 0, 1, 2, ...
    """
    # Check if sex-like
    upper = [v.strip().upper() for v in values]
    male_codes = {"M", "MALE", "1"}
    female_codes = {"F", "FEMALE", "0"}
    all_codes = male_codes | female_codes

    if all(v in all_codes for v in upper):
        return [1.0 if v in male_codes else 0.0 for v in upper]

    # General categorical encoding
    unique = sorted(set(values))
    mapping = {v: float(i) for i, v in enumerate(unique)}
    return [mapping[v] for v in values]


def _handle_missing_values(
    values: List[Optional[str]],
    strategy: str = "mean"
) -> List[float]:
    """Handle missing values in numeric covariate.

    Parameters
    ----------
    values : list
        Raw string values, may contain None or empty strings
    strategy : str
        Imputation strategy: "mean" or "median"

    Returns
    -------
    list of float
        Imputed numeric values
    """
    parsed = []
    missing_idx = []

    for i, v in enumerate(values):
        if v is None or v.strip() == "" or v.strip().upper() in ("NA", "N/A", "NAN"):
            parsed.append(None)
            missing_idx.append(i)
        else:
            parsed.append(float(v))

    if missing_idx and any(p is not None for p in parsed):
        valid = [p for p in parsed if p is not None]
        if strategy == "median":
            valid_sorted = sorted(valid)
            n = len(valid_sorted)
            fill = (valid_sorted[n // 2] + valid_sorted[(n - 1) // 2]) / 2.0
        else:
            fill = sum(valid) / len(valid)

        for i in missing_idx:
            parsed[i] = fill

    return [p if p is not None else 0.0 for p in parsed]


def _is_numeric(values: List[str]) -> bool:
    """Check if all non-missing values are numeric."""
    for v in values:
        v = v.strip()
        if v == "" or v.upper() in ("NA", "N/A", "NAN"):
            continue
        try:
            float(v)
        except ValueError:
            return False
    return True


def _load_delimited(
    path: str,
    covariate_columns: List[str],
    subject_column: str = "participant_id",
    subject_ids: Optional[List[str]] = None,
    delimiter: str = "\t",
    missing_strategy: str = "mean"
) -> ParticipantData:
    """Load demographics from a delimited file.

    Parameters
    ----------
    path : str
        Path to TSV/CSV file
    covariate_columns : list of str
        Column names to extract as covariates
    subject_column : str
        Column name for subject identifiers
    subject_ids : list of str, optional
        If provided, reorder rows to match this list.
        Subjects not found in the file raise ValueError.
    delimiter : str
        Column delimiter
    missing_strategy : str
        How to impute missing values: "mean" or "median"

    Returns
    -------
    ParticipantData
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Participants file not found: {path}")

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        fieldnames = reader.fieldnames or []

        if subject_column not in fieldnames:
            raise ValueError(
                f"Subject column '{subject_column}' not found. "
                f"Available: {fieldnames}"
            )

        for col in covariate_columns:
            if col not in fieldnames:
                raise ValueError(
                    f"Covariate column '{col}' not found. "
                    f"Available: {fieldnames}"
                )

        # Read all rows
        rows = list(reader)

    # Build lookup by subject id
    row_by_subject = {}
    for row in rows:
        sid = row[subject_column].strip()
        row_by_subject[sid] = row

    # Determine row order
    if subject_ids is not None:
        ordered_ids = []
        for sid in subject_ids:
            sid_clean = sid.strip()
            if sid_clean not in row_by_subject:
                raise ValueError(
                    f"Subject '{sid_clean}' not found in {path}. "
                    f"Available: {sorted(row_by_subject.keys())}"
                )
            ordered_ids.append(sid_clean)
    else:
        ordered_ids = [row[subject_column].strip() for row in rows]

    # Extract covariates
    covariates = {}
    for col in covariate_columns:
        raw_values = [row_by_subject[sid][col] for sid in ordered_ids]

        if _is_numeric(raw_values):
            covariates[col] = _handle_missing_values(raw_values, missing_strategy)
        else:
            covariates[col] = _encode_categorical(raw_values)

    return ParticipantData(subject_ids=ordered_ids, covariates=covariates)


def load_participants_tsv(
    path: str,
    covariate_columns: List[str],
    subject_column: str = "participant_id",
    subject_ids: Optional[List[str]] = None,
    missing_strategy: str = "mean"
) -> ParticipantData:
    """Load demographics from a BIDS participants.tsv file.

    Parameters
    ----------
    path : str
        Path to participants.tsv
    covariate_columns : list of str
        Column names to extract (e.g., ["age", "sex"])
    subject_column : str
        Column name for subject IDs (default: "participant_id")
    subject_ids : list of str, optional
        Reorder to match this list of subject IDs
    missing_strategy : str
        Imputation strategy: "mean" or "median"

    Returns
    -------
    ParticipantData
    """
    return _load_delimited(
        path, covariate_columns, subject_column, subject_ids,
        delimiter="\t", missing_strategy=missing_strategy
    )


def load_participants_csv(
    path: str,
    covariate_columns: List[str],
    subject_column: str = "subject",
    subject_ids: Optional[List[str]] = None,
    delimiter: str = ",",
    missing_strategy: str = "mean"
) -> ParticipantData:
    """Load demographics from a CSV file.

    Parameters
    ----------
    path : str
        Path to CSV file
    covariate_columns : list of str
        Column names to extract
    subject_column : str
        Column name for subject IDs (default: "subject")
    subject_ids : list of str, optional
        Reorder to match this list
    delimiter : str
        Column delimiter (default: ",")
    missing_strategy : str
        Imputation strategy: "mean" or "median"

    Returns
    -------
    ParticipantData
    """
    return _load_delimited(
        path, covariate_columns, subject_column, subject_ids,
        delimiter=delimiter, missing_strategy=missing_strategy
    )
