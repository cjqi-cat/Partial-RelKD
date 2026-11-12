#!/usr/bin/env python3
from _delegate import run_entry

run_entry(
    "stage6_7_partial_hybrid/runner/train.py",
    "Train Partial-RelKD with a frozen clean teacher and shared robust student.",
)
