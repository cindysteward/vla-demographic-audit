"""
Reads audit_results.csv and produces a bar chart of the two outcomes that
showed a signal, mahalanobis_score and n_steps, by recipient clause, with
BH-significant conditions marked. 
"""

from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt

import demoparity
from run_audit import bh_correct_all

CLAUSE_ORDER = [
    "", " for the customer", " for the visitor",
    " for the doctor", " for the nurse",
    " for the engineer", " for the receptionist",
    " for the ceo", " for the secretary",
]
CLAUSE_LABELS = [
    "none", "customer", "visitor",
    "doctor", "nurse",
    "engineer", "receptionist",
    "ceo", "secretary",
]
NEUTRAL = {"", " for the customer", " for the visitor"}


def make_figure(csv_path: str = "audit_results.csv", out_path: str = "audit_figure.png"):
    df = pd.read_csv(csv_path, keep_default_na=False)

    report = demoparity.run_audit(
        df,
        outcomes=["success_int", "mahalanobis_score", "n_steps"],
        predictors=["recipient_clause"],
        reference_levels={"recipient_clause": ""},
        alpha=0.05,
    )
    sig = bh_correct_all(report)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, outcome, ylabel in zip(
        axes, ["mahalanobis_score", "n_steps"], ["Mahalanobis distance", "Steps to completion"]
    ):
        means = df.groupby("recipient_clause")[outcome].mean().reindex(CLAUSE_ORDER)
        sems = df.groupby("recipient_clause")[outcome].sem().reindex(CLAUSE_ORDER)

        sig_terms = set(
            sig.loc[(sig["outcome"] == outcome) & (sig["significant_bh"]), "term"]
        )
        colors = [
            "#4c72b0" if clause in NEUTRAL
            else "#c44e52" if f"recipient_clause[{clause}]" in sig_terms
            else "#999999"
            for clause in CLAUSE_ORDER
        ]

        ax.bar(CLAUSE_LABELS, means.values, yerr=sems.values, color=colors, capsize=3)
        ax.axhline(means[""], linestyle="--", color="black", linewidth=0.8, alpha=0.6)
        ax.set_ylabel(ylabel)
        ax.set_title(outcome)
        ax.tick_params(axis="x", rotation=45)

    fig.suptitle(
        "OpenVLA action shift by recipient clause, red = significant vs. baseline (BH-corrected)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved {out_path}")


if __name__ == "__main__":
    make_figure()