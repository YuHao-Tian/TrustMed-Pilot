import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def ece(y, p, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(p, bins) - 1
    ece_val = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        acc = y[mask].mean()
        conf = p[mask].mean()
        ece_val += (mask.sum() / len(y)) * abs(acc - conf)
    return float(ece_val)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="processed dataset.csv")
    ap.add_argument("--preds", type=str, required=True, help="artifacts/test_predictions.csv")
    ap.add_argument("--out_dir", type=str, default="artifacts")
    return ap.parse_args()

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data)
    preds = pd.read_csv(args.preds)

    # ===== Calibration on test preds (simple diagnostics) =====
    y = preds["y"].astype(int).values
    p = preds["p"].astype(float).values

    ece_val = ece(y, p)
    brier = float(brier_score_loss(y, p))
    auroc = float(roc_auc_score(y, p))
    print(f"[CAL] AUROC={auroc:.4f} Brier={brier:.4f} ECE={ece_val:.4f}")

    frac_pos, mean_pred = calibration_curve(y, p, n_bins=15, strategy="uniform")
    plt.figure()
    plt.plot(mean_pred, frac_pos, marker="o")
    plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Reliability diagram")
    plt.tight_layout()
    plt.savefig(out_dir / "reliability.png", dpi=160)

    # ===== Selective deferral curve =====
    conf = np.maximum(p, 1-p)  # confidence as max prob
    order = np.argsort(-conf)  # high conf first
    y_sorted = y[order]
    p_sorted = p[order]
    conf_sorted = conf[order]

    coverages = np.linspace(0.1, 1.0, 10)
    accs = []
    for c in coverages:
        k = int(len(y_sorted) * c)
        if k <= 0:
            continue
        pred = (p_sorted[:k] >= 0.5).astype(int)
        acc = (pred == y_sorted[:k]).mean()
        accs.append(acc)

    plt.figure()
    plt.plot(coverages[:len(accs)], accs, marker="o")
    plt.xlabel("Coverage (kept)")
    plt.ylabel("Accuracy on kept")
    plt.title("Selective prediction (deferral) curve")
    plt.tight_layout()
    plt.savefig(out_dir / "deferral_curve.png", dpi=160)

    # ===== Membership inference baseline (loss-based) =====
    # Build a simple attacker using "loss" as feature.
    # We'll approximate per-sample loss on train vs holdout by retraining a quick LR on groups split.
    feat_cols = ["age","gender_M","ed_hist","ip_hist","op_hist","total_hist","last_gap_days"]
    X = df[feat_cols].values
    y_all = df["y"].astype(int).values
    pat = df["PATIENT"].astype(str).values

    # patient-level split for target model
    unique_p = np.unique(pat)
    train_p, test_p = train_test_split(unique_p, test_size=0.2, random_state=42)
    in_train = np.isin(pat, train_p)

    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    target = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=2000, class_weight="balanced"))])
    target.fit(X[in_train], y_all[in_train])

    prob = target.predict_proba(X)[:,1]
    eps = 1e-8
    loss = -(y_all*np.log(prob+eps) + (1-y_all)*np.log(1-prob+eps))  # log loss

    # attacker: predict membership from loss (lower loss => more likely member)
    mia_y = in_train.astype(int)
    mia_score = -loss  # higher score => more likely member
    mia_auc = float(roc_auc_score(mia_y, mia_score))
    print(f"[MIA] attack_AUC={mia_auc:.4f} (loss-based baseline)")

    # "Simple mitigation study": stronger regularization
    target_reg = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=2000, class_weight='balanced', C=0.3))])
    target_reg.fit(X[in_train], y_all[in_train])
    prob2 = target_reg.predict_proba(X)[:,1]
    loss2 = -(y_all*np.log(prob2+eps) + (1-y_all)*np.log(1-prob2+eps))
    mia_auc2 = float(roc_auc_score(mia_y, -loss2))
    print(f"[MIA] attack_AUC_after_reg={mia_auc2:.4f}")

    with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"AUROC={auroc}\nBrier={brier}\nECE={ece_val}\nMIA_AUC={mia_auc}\nMIA_AUC_reg={mia_auc2}\n")

    print(f"[OK] saved plots + summary to {out_dir}")

if __name__ == "__main__":
    main()
