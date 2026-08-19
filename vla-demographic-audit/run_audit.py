"""
build design -> run trials -> parse -> score trajectories with refstat ->
regress with demoparity -> BH-correct
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

import demoparity
import refstat

from build_design import build_design


def parse_responses(design_with_responses: pd.DataFrame) -> pd.DataFrame:
    parsed = design_with_responses["response"].apply(json.loads).apply(pd.Series)
    parsed = parsed.rename(columns={"trial_id": "traj_trial_id"})
    parsed = parsed.drop(columns=["recipient_clause"], errors="ignore")
    df = pd.concat([design_with_responses.drop(columns=["response"]), parsed], axis=1)
    df["success_int"] = df["success"].astype(int)
    return df


def score_trajectories(df: pd.DataFrame) -> pd.DataFrame:
    features = []
    for _, row in df.iterrows():
        traj = np.load(row["traj_path"])
        wv = traj["world_vector"]
        final_pos = wv.sum(axis=0)
        path_length = np.linalg.norm(np.diff(wv, axis=0), axis=1).sum() if len(wv) > 1 else 0.0
        features.append({
            "trial_id": row["trial_id"], "final_dx": final_pos[0], "final_dy": final_pos[1],
            "final_dz": final_pos[2], "path_length": path_length,
        })
    feat_df = pd.DataFrame(features).set_index("trial_id")
    df = df.set_index("trial_id").join(feat_df).reset_index()

    feature_cols = ["final_dx", "final_dy", "final_dz", "path_length", "n_steps"]
    reference_mask = df["recipient_clause"].isin({"", " for the customer", " for the visitor"})
    scorer = refstat.MahalanobisScorer()
    scorer.fit(df.loc[reference_mask, feature_cols].to_numpy())
    df["mahalanobis_score"] = df[feature_cols].apply(lambda r: scorer.score(r.to_numpy()), axis=1)
    return df


def bh_correct_all(report, alpha: float = 0.05) -> pd.DataFrame:
    rows = []
    for outcome, model in report.models.items():
        for term, p in model.pvalues.items():
            if term == "Intercept":
                continue
            rows.append({"outcome": outcome, "term": term, "p_raw": p, "coef": model.params[term]})
    result = pd.DataFrame(rows)
    reject, p_adj, _, _ = multipletests(result["p_raw"], alpha=alpha, method="fdr_bh")
    result["p_bh"] = p_adj
    result["significant_bh"] = reject
    return result.sort_values("p_bh")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--real", action="store_true")
    args = parser.parse_args()
    if args.mock == args.real:
        raise SystemExit("pass exactly one of --mock or --real")

    design = build_design()
    if args.mock:
        from mock_generator import make_generator
    else:
        from vla_generator import make_generator
    generator = make_generator()

    results = demoparity.run_design(design, generator, concurrency=1, checkpoint_path="./checkpoint.jsonl")
    df = parse_responses(results)
    df = score_trajectories(df)
    df.to_csv("./audit_results.csv", index=False)

    report = demoparity.run_audit(
        df,
        outcomes=["success_int", "mahalanobis_score", "n_steps"],
        predictors=["recipient_clause"],
        reference_levels={"recipient_clause": ""},
        alpha=0.05,
    )

    print("\n=== descriptive stats ===")
    print(report.descriptives)

    print("\n=== BH-corrected results, family-wise, this is the table that matters ===")
    corrected = bh_correct_all(report)
    print(corrected.to_string(index=False))

    print("\nfull results written to audit_results.csv")


if __name__ == "__main__":
    main()
