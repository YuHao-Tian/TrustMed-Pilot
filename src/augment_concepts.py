import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import re

def to_dt_utc_naive(s):
    # robust: parse any timezone-aware/naive, unify to UTC then drop tz
    dt = pd.to_datetime(s, errors="coerce", utc=True)
    # dt is tz-aware in UTC; drop tz to avoid tz-aware vs tz-naive subtraction issues
    return dt.dt.tz_convert(None)

def earliest_date_by_patient(df, patient_col, date_col, mask):
    sub = df.loc[mask, [patient_col, date_col]].dropna()
    if sub.empty:
        return pd.Series(dtype="datetime64[ns]")
    return sub.groupby(patient_col)[date_col].min()

def earliest_threshold_date_obs(obs, desc_regex, threshold):
    # returns Series: patient -> earliest DATE where obs matches regex and VALUE >= threshold
    m = obs["DESCRIPTION"].fillna("").str.contains(desc_regex, case=False, regex=True, na=False)
    sub = obs.loc[m, ["PATIENT", "DATE", "VALUE"]].copy()
    if sub.empty:
        return pd.Series(dtype="datetime64[ns]")
    sub["VALUE_NUM"] = pd.to_numeric(sub["VALUE"], errors="coerce")
    sub = sub.dropna(subset=["DATE", "VALUE_NUM"])
    sub = sub.loc[sub["VALUE_NUM"] >= threshold, ["PATIENT", "DATE"]]
    if sub.empty:
        return pd.Series(dtype="datetime64[ns]")
    return sub.groupby("PATIENT")["DATE"].min()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_dir", type=str, required=True, help="Synthea output/csv directory")
    ap.add_argument("--in_dataset", type=str, required=True, help="processed dataset.csv")
    ap.add_argument("--out_dataset", type=str, required=True, help="output dataset_v2.csv")
    args = ap.parse_args()

    csv_dir = Path(args.csv_dir)
    df = pd.read_csv(args.in_dataset)

    df["t_dt"] = pd.to_datetime(df["t"], errors="coerce", utc=True).dt.tz_convert(None)

    cond = pd.read_csv(csv_dir / "conditions.csv")
    meds = pd.read_csv(csv_dir / "medications.csv")
    obs  = pd.read_csv(csv_dir / "observations.csv")

    if "START" in cond.columns:
        cond["START"] = to_dt_utc_naive(cond["START"])
    if "START" in meds.columns:
        meds["START"] = to_dt_utc_naive(meds["START"])
    if "DATE" in obs.columns:
        obs["DATE"] = to_dt_utc_naive(obs["DATE"])

    cond_desc = cond["DESCRIPTION"].fillna("")
    meds_desc = meds["DESCRIPTION"].fillna("")
    obs_desc  = obs["DESCRIPTION"].fillna("")

    # --------- Define concept rules----------
    concept_rules_conditions = {
        "c_htn": r"\bhypertension\b",
        "c_diabetes": r"\bdiabetes\b",
        "c_hyperlipidemia": r"hyperlipidemia|hypercholesterolemia",
        "c_obesity": r"\bobesity\b",
        "c_asthma_copd": r"\basthma\b|copd|chronic obstructive",
        "c_ckd": r"chronic kidney|renal failure|kidney disease",
        "c_cad_heart": r"coronary|ischemic heart|myocard|angina|heart failure",
        "c_stroke": r"\bstroke\b|cerebral infarction|cerebrovascular",
        "c_cancer": r"cancer|carcinoma|malignan",
        "c_chronic_pain": r"chronic pain|back pain|osteoarthritis|fibromyalgia|pain",
        "c_sleep_disorder": r"sleep",
        "c_depression_hist": r"depression|major depressive",
        "c_anxiety_hist": r"anxiety|panic",
        "c_substance_use": r"substance|drug abuse|opioid dependence|cannabis|cocaine",
        "c_alcohol_use": r"alcohol",
    }

    concept_rules_meds = {
        "c_antidepressant": r"sertraline|fluoxetine|citalopram|escitalopram|paroxetine|venlafaxine|duloxetine|bupropion|mirtazapine",
        "c_benzo": r"diazepam|lorazepam|alprazolam|clonazepam",
        "c_opioid_rx": r"oxycodone|hydrocodone|morphine|fentanyl|codeine|tramadol",
        "c_statin_rx": r"atorvastatin|simvastatin|rosuvastatin|pravastatin",
    }

    obs_thresholds = {
        "c_high_bmi": (r"body mass index|BMI", 30.0),
        "c_high_sbp": (r"systolic blood pressure", 140.0),
        "c_high_dbp": (r"diastolic blood pressure", 90.0),
        "c_high_a1c": (r"hemoglobin a1c|HbA1c", 6.5),
    }

    # --------- Precompute earliest event dates per patient for each concept ----------
    earliest_dates = {}

    # conditions concepts
    for cname, regex in concept_rules_conditions.items():
        mask = cond_desc.str.contains(regex, case=False, regex=True, na=False)
        earliest_dates[cname] = earliest_date_by_patient(cond, "PATIENT", "START", mask)

    # meds concepts
    for cname, regex in concept_rules_meds.items():
        mask = meds_desc.str.contains(regex, case=False, regex=True, na=False)
        earliest_dates[cname] = earliest_date_by_patient(meds, "PATIENT", "START", mask)

    # obs thresholds
    for cname, (regex, thr) in obs_thresholds.items():
        earliest_dates[cname] = earliest_threshold_date_obs(obs, regex, thr)

    # --------- Attach concepts to dataset as-of time t ----------
    for cname, s in earliest_dates.items():
        dmap = s.to_dict()
        ev = df["PATIENT"].map(dmap)
        ev = pd.to_datetime(ev, errors="coerce")
        df[cname] = ((~ev.isna()) & (df["t_dt"] >= ev)).astype(int)

    try:
        meds2 = meds.loc[~meds["START"].isna(), ["PATIENT", "START", "DESCRIPTION"]].copy()
        meds2["DESCRIPTION"] = meds2["DESCRIPTION"].fillna("")
        meds2 = meds2.sort_values(["PATIENT", "START"])
        meds2["desc_norm"] = meds2["DESCRIPTION"].str.lower()
        meds2["is_new"] = meds2.groupby(["PATIENT", "desc_norm"]).cumcount() == 0
        meds2["cum_distinct"] = meds2.groupby("PATIENT")["is_new"].cumsum()

        # For each patient and each row time, need last cum_distinct at or before t
        df["c_polypharmacy5"] = 0
        for pid, g in df.groupby("PATIENT", sort=False):
            m = meds2.loc[meds2["PATIENT"] == pid, ["START", "cum_distinct"]]
            if m.empty:
                continue
            m = m.sort_values("START")
            tg = g[["t_dt"]].copy()
            tg = tg.sort_values("t_dt")
            idx = np.searchsorted(m["START"].values.astype("datetime64[ns]"),
                                  tg["t_dt"].values.astype("datetime64[ns]"),
                                  side="right") - 1
            last_cnt = np.where(idx >= 0, m["cum_distinct"].values[idx], 0)
            df.loc[tg.index, "c_polypharmacy5"] = (last_cnt >= 5).astype(int)
    except Exception as e:
        print("[WARN] polypharmacy concept skipped:", e)

    df = df.drop(columns=["t_dt"], errors="ignore")

    out_path = Path(args.out_dataset)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[OK] wrote {out_path} with {len(df.columns)} columns")

if __name__ == "__main__":
    main()
