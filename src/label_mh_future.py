import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def to_dt_utc_naive(s):
    dt = pd.to_datetime(s, errors="coerce", utc=True)
    return dt.dt.tz_convert(None)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_dir", type=str, required=True, help="Synthea output/csv dir")
    ap.add_argument("--in_dataset", type=str, required=True, help="dataset_v2.csv")
    ap.add_argument("--out_dataset", type=str, required=True, help="dataset_mh.csv")
    ap.add_argument("--horizon_days", type=int, default=365)
    ap.add_argument("--target", type=str, default="both", choices=["depression", "anxiety", "both"])
    ap.add_argument("--incident_only", action="store_true", help="drop rows with prior mh history before t")
    args = ap.parse_args()

    csv_dir = Path(args.csv_dir)
    df = pd.read_csv(args.in_dataset)
    df["t_dt"] = pd.to_datetime(df["t"], errors="coerce", utc=True).dt.tz_convert(None)

    cond = pd.read_csv(csv_dir / "conditions.csv")
    cond["START"] = to_dt_utc_naive(cond["START"])
    desc = cond["DESCRIPTION"].fillna("").str.lower()

    dep_regex = r"depression|major depressive"
    anx_regex = r"anxiety|panic"

    if args.target == "depression":
        m = desc.str.contains(dep_regex, regex=True, na=False)
    elif args.target == "anxiety":
        m = desc.str.contains(anx_regex, regex=True, na=False)
    else:
        m = desc.str.contains(dep_regex, regex=True, na=False) | desc.str.contains(anx_regex, regex=True, na=False)

    mh = cond.loc[m, ["PATIENT", "START"]].dropna()
    mh = mh.sort_values(["PATIENT", "START"])

    # build patient -> sorted start dates
    mh_map = {}
    for pid, g in mh.groupby("PATIENT"):
        mh_map[pid] = g["START"].values.astype("datetime64[ns]")

    H = np.timedelta64(args.horizon_days, "D")

    y = np.zeros(len(df), dtype=int)
    prior = np.zeros(len(df), dtype=int)

    # process per patient for speed
    for pid, idx in df.groupby("PATIENT").groups.items():
        dates = mh_map.get(pid)
        if dates is None or len(dates) == 0:
            continue
        tvals = df.loc[idx, "t_dt"].values.astype("datetime64[ns]")
        pos = np.searchsorted(dates, tvals, side="right")
        prior[idx] = (pos > 0).astype(int)
        has_next = pos < len(dates)
        next_date = np.full(len(tvals), np.datetime64("NaT"), dtype="datetime64[ns]")
        next_date[has_next] = dates[pos[has_next]]

        y[idx] = (has_next & (next_date <= (tvals + H))).astype(int)

    out = df.copy()
    out["y_prev"] = out["y"]
    out["mh_prior"] = prior
    out["y"] = y
    out = out.drop(columns=["t_dt"], errors="ignore")

    if args.incident_only:
        out = out.loc[out["mh_prior"] == 0].reset_index(drop=True)

    out_path = Path(args.out_dataset)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print("[OK] wrote", out_path)
    print("[INFO] y prevalence =", float(out["y"].mean()), "n =", len(out))

if __name__ == "__main__":
    main()
