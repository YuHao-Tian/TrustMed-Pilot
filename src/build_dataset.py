import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_dir", type=str, required=True, help=".../synthea/csv")
    ap.add_argument("--out", type=str, default="data/processed/dataset.csv")
    ap.add_argument("--lookback_days", type=int, default=180)
    ap.add_argument("--horizon_days", type=int, default=180)
    ap.add_argument("--min_history", type=int, default=2)
    return ap.parse_args()

def main():
    args = parse_args()
    csv_dir = Path(args.csv_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    patients = pd.read_csv(csv_dir / "patients.csv")
    enc = pd.read_csv(csv_dir / "encounters.csv")

    # Basic cleanup
    patients = patients.rename(columns={"Id": "PATIENT"})
    enc["START"] = pd.to_datetime(enc["START"], errors="coerce", utc=True).dt.tz_localize(None)
    enc["STOP"] = pd.to_datetime(enc["STOP"], errors="coerce", utc=True).dt.tz_localize(None)
    enc = enc.dropna(subset=["PATIENT", "START"])

    # demographic features
    patients["BIRTHDATE"] = pd.to_datetime(patients["BIRTHDATE"], errors="coerce").dt.tz_localize(None)
    patients["GENDER"] = patients["GENDER"].fillna("U")

    pat_map = patients.set_index("PATIENT")[["BIRTHDATE", "GENDER"]].to_dict(orient="index")

    if "ENCOUNTERCLASS" not in enc.columns:
        raise ValueError("encounters.csv missing ENCOUNTERCLASS column")

    enc = enc.sort_values(["PATIENT", "START"])

    rows = []
    lookback = pd.Timedelta(days=args.lookback_days)
    horizon = pd.Timedelta(days=args.horizon_days)

    for pid, g in enc.groupby("PATIENT"):
        g = g.sort_values("START").reset_index(drop=True)
        if len(g) < args.min_history:
            continue

        demo = pat_map.get(pid, None)
        if demo is None:
            continue

        birth = demo["BIRTHDATE"]
        gender = demo["GENDER"]

        starts = g["START"].values
        classes = g["ENCOUNTERCLASS"].astype(str).values

        # iterate each encounter as an index point
        for i in range(args.min_history, len(g) - 1):
            t = g.loc[i, "START"]
            lb_start = t - lookback
            hz_end = t + horizon

            hist = g[(g["START"] >= lb_start) & (g["START"] < t)]
            fut = g[(g["START"] > t) & (g["START"] <= hz_end)]

            if len(hist) < args.min_history:
                continue

            # label: any ED in horizon
            y = int((fut["ENCOUNTERCLASS"].astype(str) == "emergency").any())

            # simple interpretable "concepts" from utilization patterns
            ed_hist = int((hist["ENCOUNTERCLASS"].astype(str) == "emergency").sum())
            ip_hist = int((hist["ENCOUNTERCLASS"].astype(str) == "inpatient").sum())
            op_hist = int((hist["ENCOUNTERCLASS"].astype(str) == "outpatient").sum())
            total_hist = int(len(hist))

            # concepts (0/1)
            c_frequent_ed = int(ed_hist >= 2)
            c_recent_inpatient = int(ip_hist >= 1)
            c_high_util = int(total_hist >= 5)

            # time features
            last_gap_days = (t - hist["START"].iloc[-1]).days if len(hist) > 0 else 999

            # age
            if pd.isna(birth):
                age = 0
            else:
                age = int((t - birth).days / 365.25)
                age = max(age, 0)

            rows.append({
                "PATIENT": pid,
                "t": t,
                "age": age,
                "gender_M": int(str(gender).upper() == "M"),
                "ed_hist": ed_hist,
                "ip_hist": ip_hist,
                "op_hist": op_hist,
                "total_hist": total_hist,
                "last_gap_days": last_gap_days,
                "c_frequent_ed": c_frequent_ed,
                "c_recent_inpatient": c_recent_inpatient,
                "c_high_util": c_high_util,
                "y": y,
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"[OK] saved dataset: {out_path} rows={len(df)} patients={df['PATIENT'].nunique()}")

if __name__ == "__main__":
    main()
