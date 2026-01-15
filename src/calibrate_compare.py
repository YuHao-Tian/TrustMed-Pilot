import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

EPS = 1e-12


def sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def nll(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def _make_bins(p, n_bins=10, strategy="uniform"):
    """
    Returns bin edges.
    - uniform: equally spaced in [0,1]
    - quantile: edges are quantiles of p (approximately equal samples per bin)
    """
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    if n_bins < 2:
        return np.array([0.0, 1.0], dtype=float)

    if strategy == "uniform":
        return np.linspace(0.0, 1.0, n_bins + 1)

    if strategy == "quantile":
        qs = np.linspace(0.0, 1.0, n_bins + 1)
        bins = np.quantile(p, qs)
        return bins

    raise ValueError(f"Unknown binning strategy: {strategy}")


def ece(y, p, n_bins=10, strategy="uniform"):
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    bins = _make_bins(p, n_bins=n_bins, strategy=strategy)

    e = 0.0
    n = len(y)
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        m = (p >= lo) & (p < hi) if i < (len(bins) - 2) else (p >= lo) & (p <= hi)
        if not np.any(m):
            continue
        acc = y[m].mean()
        conf = p[m].mean()
        e += abs(acc - conf) * (m.sum() / n)
    return float(e)


def reliability_points(y, p, n_bins=10, strategy="uniform"):
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    bins = _make_bins(p, n_bins=n_bins, strategy=strategy)

    xs, ys = [], []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        m = (p >= lo) & (p < hi) if i < (len(bins) - 2) else (p >= lo) & (p <= hi)
        if not np.any(m):
            continue
        xs.append(p[m].mean())
        ys.append(y[m].mean())
    return np.array(xs), np.array(ys)


def plot_reliability(y, p, title, out_path, n_bins=10, strategy="quantile", auto_xlim=True):
    xs, ys = reliability_points(y, p, n_bins=n_bins, strategy=strategy)

    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    q99 = float(np.quantile(p, 0.99))

    plt.figure()
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.plot(xs, ys, marker="o")
    plt.title(title)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")

    # If probabilities are very small (common after Platt when prevalence is low),
    # show a zoomed-in x-axis so the curve is readable.
    if auto_xlim and q99 < 0.2:
        right = min(0.2, max(0.05, q99 * 1.2))
        plt.xlim(0.0, right)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_deferral(y, p, title, out_path):
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)

    conf = np.maximum(p, 1 - p)
    order = np.argsort(-conf)
    y = y[order]
    p = p[order]
    coverages = np.linspace(0.1, 1.0, 10)
    accs = []
    for c in coverages:
        k = max(1, int(round(len(y) * c)))
        pred = (p[:k] >= 0.5).astype(int)
        accs.append((pred == y[:k]).mean())

    plt.figure()
    plt.plot(coverages, accs, marker="o")
    plt.title(title)
    plt.xlabel("Coverage (kept)")
    plt.ylabel("Accuracy on kept")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def fit_temperature(logit_val, y_val):
    Ts = np.logspace(-1, np.log10(10), 200)
    best_T, best_nll = 1.0, float("inf")
    for T in Ts:
        p = sigmoid(logit_val / T)
        cur = nll(y_val, p)
        if cur < best_nll:
            best_nll = cur
            best_T = float(T)
    return best_T


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=str, required=True)
    ap.add_argument("--bins", type=int, default=10)
    # Keep ECE standard by default; make plots readable by default
    ap.add_argument("--ece_binning", choices=["uniform", "quantile"], default="uniform")
    ap.add_argument("--plot_binning", choices=["uniform", "quantile"], default="quantile")
    ap.add_argument("--no_auto_xlim", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    art = Path(args.artifacts)

    val = pd.read_csv(art / "val_predictions.csv")
    te = pd.read_csv(art / "test_predictions.csv")

    for need in ["y", "p", "logit"]:
        if need not in val.columns or need not in te.columns:
            raise ValueError(
                f"Missing column '{need}' in predictions. Re-run train_baseline.py to export it."
            )

    yv = val["y"].astype(int).values
    yt = te["y"].astype(int).values
    pv = val["p"].astype(float).values
    pt = te["p"].astype(float).values
    lv = val["logit"].astype(float).values
    lt = te["logit"].astype(float).values

    results = []

    def eval_and_save(label, p_test, tag):
        res = {
            "name": label,
            "AUROC": float(roc_auc_score(yt, p_test)),
            "AUPRC": float(average_precision_score(yt, p_test)),
            "Brier": float(brier_score_loss(yt, p_test)),
            "ECE": ece(yt, p_test, n_bins=args.bins, strategy=args.ece_binning),
            "NLL": nll(yt, p_test),
        }
        results.append(res)
        plot_reliability(
            yt,
            p_test,
            f"Reliability ({label})",
            art / f"reliability_{tag}.png",
            n_bins=args.bins,
            strategy=args.plot_binning,
            auto_xlim=(not args.no_auto_xlim),
        )
        plot_deferral(yt, p_test, f"Deferral ({label})", art / f"deferral_{tag}.png")
        return res

    # before
    eval_and_save("before", pt, "before")

    # temperature scaling
    T = fit_temperature(lv, yv)
    pt_temp = sigmoid(lt / T)
    eval_and_save(f"temp(T={T:.2f})", pt_temp, "temp")

    # platt scaling
    platt = LogisticRegression(solver="lbfgs", max_iter=1000)
    platt.fit(lv.reshape(-1, 1), yv)
    pt_platt = platt.predict_proba(lt.reshape(-1, 1))[:, 1]
    eval_and_save("platt", pt_platt, "platt")

    # isotonic
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(pv, yv)
    pt_iso = iso.transform(pt)
    eval_and_save("isotonic", pt_iso, "isotonic")

    summary_path = art / "calibration_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(str(r) + "\n")

    print(f"[OK] wrote {summary_path} + plots to {art}")


if __name__ == "__main__":
    main()
