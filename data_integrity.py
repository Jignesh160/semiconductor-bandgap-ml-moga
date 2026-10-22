"""Small data-integrity helpers for leakage-safe material evaluation."""

from sklearn.model_selection import train_test_split


def split_real_records(records, test_size=0.2, random_state=42, id_column="jid"):
    """Split verified records once and enforce unique, disjoint material IDs."""
    if id_column not in records:
        raise ValueError(f"Required provenance column {id_column!r} is missing.")
    raw_ids = records[id_column]
    ids = raw_ids.astype(str)
    if raw_ids.isna().any() or ids.eq("").any() or ids.duplicated().any():
        raise ValueError(f"{id_column!r} values must be present and unique.")
    train, test = train_test_split(
        records, test_size=test_size, random_state=random_state,
    )
    if set(train[id_column]) & set(test[id_column]):
        raise RuntimeError("Material ID overlap detected between train and test.")
    return train, test
