#!/usr/bin/env python3
"""
Matching Track validation script
"""
from argparse import ArgumentParser, Namespace
from typing import List

import pandas as pd

parser = ArgumentParser()
parser.add_argument(
    "--path",
    help="Path to match csv",
    type=str,
    required=True,
)


class DataValidationError(AssertionError):
    pass


def validate_column_order(columns: List[str]):
    expected_cols = [
        "query_id",
        "ref_id",
        "query_start",
        "query_end",
        "ref_start",
        "ref_end",
        "score",
    ]

    for i, (expected, found) in enumerate(zip(expected_cols, columns)):
        if expected != found:
            raise DataValidationError(
                f"Columns in incorrect order. Expected {expected} in position {i}, but got {found}."
            )


def validate_timestamps(df: pd.DataFrame):
    for col in ["query_start", "query_end", "ref_start", "ref_end"]:
        if not (df[col] >= 0).all():
            raise DataValidationError(
                f"Found negative timestamps in {col}: "
                f"all timestamps should be greater than or equal to zero."
            )
    if not (df["query_start"] <= df["query_end"]).all():
        raise DataValidationError(
            f"Found query start timestamps greater than query end timestamps: "
            f"all end timestamps should be greater than start timestamps."
        )
    if not (df["ref_start"] <= df["ref_end"]).all():
        raise DataValidationError(
            f"Found query start timestamps greater than query end timestamps: "
            f"all end timestamps should be greater than start timestamps."
        )


def main(args: Namespace):
    matches_df = pd.read_csv(args.path)
    validate_column_order(matches_df.columns)
    validate_timestamps(matches_df)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
