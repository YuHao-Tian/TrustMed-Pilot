import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, brier_score_loss


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--splits", type=str, required=True, help="CSV with columns: row_id, split (train/val/test)")
    ap.add_argument("--out_dir", type=str, default="artifacts")
    ap.add_argument("--exclude_util_concepts", action="store_true",
                    help="Exclude util concepts: c_frequent_ed, c_recent_inpatient, c_high_util")
    return ap.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    df.columns = df.columns.str.strip()
    df = df.reset_index().rename(columns={"index": "row_id"})
    if "y" not in df.columns:
        raise ValueError("dataset must contain column 'y'")

    sp = pd.read_csv(args.splits)
    sp.columns = sp.columns.str.strip()
    if "row_id" not in sp.columns or "split" not in sp.columns:
        raise ValueError("splits.csv must contain columns: row_id, split")

    # masks
    tr_ids = set(sp.loc[sp["split"] == "train", "row_id"].astype(int).tolist())
    va_ids = set(sp.loc[sp["split"] == "val", "row_id"].astype(int).tolist())
    te_ids = set(sp.loc[sp["split"] == "test", "row_id"].astype(int).tolist())

    tr = df["row_id"].isin(tr_ids).values
    va = df["row_id"].isin(va_ids).values
    te = df["row_id"].isin(te_ids).values

    y = df["y"].astype(int).values
    ytr, yva, yte = y[tr], y[va], y[te]

    # base features for concept prediction
    feat_cols = ["age", "gender_M", "ed_hist", "ip_hist", "op_hist", "total_hist", "last_gap_days"]
    feat_cols = [c for c in feat_cols if c in df.columns]
    X = df[feat_cols].values.astype(float)
    Xtr, Xva, Xte = X[tr], X[va], X[te]

    # concept columns
    concept_cols = [c for c in df.columns if c.startswith("c_")]
    util_concepts = {"c_frequent_ed", "c_recent_inpatient", "c_high_util"}
    if args.exclude_util_concepts:
        concept_cols = [c for c in concept_cols if c not in util_concepts]

    if len(concept_cols) == 0:
        raise ValueError("No concept columns found (c_*) after filtering.")

    used_concepts = []
    skipped_single_class = []

    # store predicted concept probabilities for label model
    Ctr_hat = []
    Cva_hat = []
    Cte_hat = []

    # also keep true concepts
    Ctr_true = df.loc[tr, concept_cols].astype(int).values
    Cte_true = df.loc[te, concept_cols].astype(int).values

    concept_acc = {}

    for c in concept_cols:
        ctr = df.loc[tr, c].astype(int).values
        cte = df.loc[te, c].astype(int).values

        uniq = np.unique(ctr)
        if len(uniq) < 2:
            skipped_single_class.append((c, uniq.tolist()))
            continue

        m = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=2000, class_weight="balanced"))
        ])
        m.fit(Xtr, ctr)

        p_tr = m.predict_proba(Xtr)[:, 1]
        p_va = m.predict_proba(Xva)[:, 1]
        p_te = m.predict_proba(Xte)[:, 1]

        pred_te = (p_te >= 0.5).astype(int)
        concept_acc[c] = float(accuracy_score(cte, pred_te))

        used_concepts.append(c)
        Ctr_hat.append(p_tr)
        Cva_hat.append(p_va)
        Cte_hat.append(p_te)

    n_total = len(concept_cols)
    n_used = len(used_concepts)
    n_skipped = len(skipped_single_class)

    if n_used == 0:
        # write summary and exit
        summary_path = out_dir / "cbm_summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"n_concepts_total = {n_total}\n")
            f.write(f"n_concepts_used  = 0\n")
            f.write(f"n_concepts_skipped(single-class-train) = {n_skipped}\n\n")
            f.write("All concepts were skipped due to single-class in train.\n")
        print(f"[OK] wrote {summary_path}")
        for c, uniq in skipped_single_class:
            print(f"[SKIP] {c}: only one class in train ({uniq})")
        return

    # stack concept features
    Ctr_hat = np.vstack(Ctr_hat).T  # [n_train, n_used]
    Cva_hat = np.vstack(Cva_hat).T
    Cte_hat = np.vstack(Cte_hat).T

    # label model trained on predicted concepts
    label_pred = LogisticRegression(max_iter=2000, class_weight="balanced")
    label_pred.fit(Ctr_hat, ytr)
    p_te_pred = label_pred.predict_proba(Cte_hat)[:, 1]

    cbm_pred = {
        "AUROC": float(roc_auc_score(yte, p_te_pred)) if len(np.unique(yte)) > 1 else float("nan"),
        "AUPRC": float(average_precision_score(yte, p_te_pred)),
        "Brier": float(brier_score_loss(yte, p_te_pred)),
    }

    # oracle upper bound: label model trained on TRUE concepts
    used_idx = [concept_cols.index(c) for c in used_concepts]
    Ctr_true_used = Ctr_true[:, used_idx]
    Cte_true_used = Cte_true[:, used_idx]

    label_oracle = LogisticRegression(max_iter=2000, class_weight="balanced")
    label_oracle.fit(Ctr_true_used, ytr)
    p_te_oracle = label_oracle.predict_proba(Cte_true_used)[:, 1]

    cbm_oracle = {
        "AUROC": float(roc_auc_score(yte, p_te_oracle)) if len(np.unique(yte)) > 1 else float("nan"),
        "AUPRC": float(average_precision_score(yte, p_te_oracle)),
        "Brier": float(brier_score_loss(yte, p_te_oracle)),
    }

    w = label_oracle.coef_.ravel()
    names = np.array(used_concepts)

    k = min(15, len(w))
    top = np.argsort(np.abs(w))[::-1][:k]
    names_top = names[top]
    w_top = w[top]

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(w_top)), w_top)
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.xticks(range(len(w_top)), names_top, rotation=25, ha="right")
    plt.ylabel("Weight")
    plt.title("CBM label model: top concept weights")
    plt.tight_layout()
    fig_path = out_dir / "cbm_concept_weights.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()

    summary_path = out_dir / "cbm_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"n_concepts_total = {n_total}\n")
        f.write(f"n_concepts_used  = {n_used}\n")
        f.write(f"n_concepts_skipped(single-class-train) = {n_skipped}\n\n")

        if n_skipped:
            f.write("Skipped concepts:\n")
            for c, uniq in skipped_single_class:
                f.write(f"  {c}: only one class in train ({uniq})\n")
            f.write("\n")

        f.write("Concept test accuracy (top 10 shown):\n")
        for c, acc in sorted(concept_acc.items(), key=lambda x: x[1], reverse=True)[:10]:
            f.write(f"  {c}: {acc:.4f}\n")

        f.write("\nCBM test metrics (label model trained on *predicted* concepts):\n")
        f.write(f"  CBM_pred AUROC: {cbm_pred['AUROC']:.4f}\n")
        f.write(f"  CBM_pred AUPRC: {cbm_pred['AUPRC']:.4f}\n")
        f.write(f"  CBM_pred Brier: {cbm_pred['Brier']:.4f}\n")

        f.write("\nOracle upper bound (label model trained on *true* concepts):\n")
        f.write(f"  CBM_oracle AUROC: {cbm_oracle['AUROC']:.4f}\n")
        f.write(f"  CBM_oracle AUPRC: {cbm_oracle['AUPRC']:.4f}\n")
        f.write(f"  CBM_oracle Brier: {cbm_oracle['Brier']:.4f}\n")

    for c, uniq in skipped_single_class:
        print(f"[SKIP] {c}: only one class in train ({uniq})")

    print(f"[OK] wrote cbm_summary.txt + cbm_concept_weights.png to {out_dir}")
    print(f"[INFO] used concepts: {n_used} / {n_total} (skipped {n_skipped})")
    print("[CBM_pred]", cbm_pred)
    print("[CBM_oracle]", cbm_oracle)


if __name__ == "__main__":
    main()
