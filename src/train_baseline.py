import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, accuracy_score


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--splits", type=str, default=None, help="CSV with columns: row_id, split")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def _ensure_row_id(df: pd.DataFrame) -> pd.DataFrame:
    if "row_id" in df.columns:
        return df
    return df.reset_index().rename(columns={"index": "row_id"})


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    df.columns = df.columns.str.strip()
    df = _ensure_row_id(df)

    if "y" not in df.columns:
        raise ValueError("Missing column 'y' in dataset.")
    y = df["y"].astype(int).values

    pid_col = "PATIENT" if "PATIENT" in df.columns else ("patient" if "patient" in df.columns else None)

    drop_cols = {"y", "row_id"}
    if pid_col:
        drop_cols.add(pid_col)
        
    feat_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values

    if args.splits:
        sp = pd.read_csv(args.splits)
        sp.columns = sp.columns.str.strip()
        if not {"row_id", "split"}.issubset(set(sp.columns)):
            raise ValueError("splits.csv must contain columns: row_id, split")
        m = df[["row_id"]].merge(sp[["row_id", "split"]], on="row_id", how="left")
        if m["split"].isna().any():
            raise ValueError("Some rows missing split assignment; check splits.csv vs dataset row_id.")
        split_arr = m["split"].values
    else:
        rng = np.random.default_rng(args.seed)
        idx = np.arange(len(df))
        rng.shuffle(idx)
        n = len(idx)
        n_tr = int(0.65 * n)
        n_va = int(0.15 * n)
        split_arr = np.array(["test"] * n, dtype=object)
        split_arr[idx[:n_tr]] = "train"
        split_arr[idx[n_tr:n_tr + n_va]] = "val"

    tr = split_arr == "train"
    va = split_arr == "val"
    te = split_arr == "test"

    Xtr, ytr = X[tr], y[tr]
    Xva, yva = X[va], y[va]
    Xte, yte = X[te], y[te]

    clf = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("lr", LogisticRegression(max_iter=2000, solver="lbfgs", class_weight="balanced")),
    ])
    clf.fit(Xtr, ytr)

    pva = clf.predict_proba(Xva)[:, 1]
    pte = clf.predict_proba(Xte)[:, 1]
    lva = clf.decision_function(Xva)   # logit
    lte = clf.decision_function(Xte)

    metrics = {
        "AUROC": float(roc_auc_score(yte, pte)),
        "AUPRC": float(average_precision_score(yte, pte)),
        "Brier": float(brier_score_loss(yte, pte)),
        "Acc@0.5": float(accuracy_score(yte, (pte >= 0.5).astype(int))),
        "n_train": int(tr.sum()),
        "n_val": int(va.sum()),
        "n_test": int(te.sum()),
        "train_prevalence": float(ytr.mean()),
        "val_prevalence": float(yva.mean()),
        "test_prevalence": float(yte.mean()),
    }
    print("[METRICS]", metrics)

    joblib.dump(clf, out_dir / "baseline_lr.joblib")

    sp_out = pd.DataFrame({"row_id": df["row_id"].values, "split": split_arr})
    if pid_col:
        sp_out[pid_col] = df[pid_col].astype(str).values
    sp_out.to_csv(out_dir / "splits.csv", index=False)

    base_cols = ["row_id", "y"] + ([pid_col] if pid_col else [])
    val_pred = df.loc[va, base_cols].copy()
    val_pred["p"] = pva
    val_pred["logit"] = lva
    val_pred.to_csv(out_dir / "val_predictions.csv", index=False)

    test_pred = df.loc[te, base_cols].copy()
    test_pred["p"] = pte
    test_pred["logit"] = lte
    test_pred.to_csv(out_dir / "test_predictions.csv", index=False)

    print(f"[OK] saved artifacts to {out_dir}")


if __name__ == "__main__":
    main()
