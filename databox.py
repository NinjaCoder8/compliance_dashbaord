#!/usr/bin/env python3
"""
Build a single-sheet, Databox-ready Excel/CSV from compliance + enrollments exports.

Inputs (in the same folder as this script by default):
  - compliance2.csv OR .xlsx  (complaints)
  - enrollments.csv  OR .xlsx  (call/enrollment logs)

Outputs:
  - databox_feed.xlsx  (sheet: 'databox')
  - databox_feed.csv   (same columns as the Excel sheet)

The sheet is column-oriented and includes one Date column plus multiple numeric metric columns,
which Databox's Excel/Sheets Metric Builder can map to separate Custom Metrics.
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd

# -------------------------
# Config / business rules
# -------------------------
# Enrollments: require connected == True (your earlier default)
REQUIRE_CONNECTED = True

# Complaints: exclude carriers marked as "(Unknown)" from ALL complaint-based calculations
EXCLUDE_UNKNOWN_CARRIERS = True

# -------------------------
# Column definitions (from your Streamlit app)
# -------------------------
DATE_COLS_CANDIDATES = [
    "Enrollment Date",
    "Date of Occurence",  # spelled as provided
    "Follow-Up Date",
    "Deadline Date",
    "Carrier Ruling Date",
    "Last edited time",
    "Date Complaint Received",
    "Complaint Received Date",
    "Date Received",
]

BOOL_COLS_CANDIDATES = [
    "CTM",
    "Spanish?",
    "Inactive Agent?",
    "Remedial Sent",
    "Remedial Signed",
    "Sales Eval Sent",
    "Sales Eval Signed",
    "Course Assigned",
    "Course Completed",
    "Completed Files",
]

NUM_COLS_CANDIDATES = [
    "Type 1 Fee %",
    "Type 2 Fee %",
    "Recording Score",
]

TEXT_COLS = [
    "Member Name",
    "Agent Name",
    "Carrier Name",
    "Case Status",
    "Case Number",
    "Case Issues Type(s)",
    "Call Issue Type(s)",
    "BLFG ID",
    "Team Name",
]

ENR_EXPECTED = [
    "agencyId","agencyName","agentEmail","agentFirstName","agentId","agentLastName","agentNpn",
    "billable","billableCostBase","billableCostMinutes","billableCostTotal","billableSeconds","callType",
    "createdAt","direction","duration","durationInCall","durationInQueue","enrollmentCode","enrollmentCodeNote",
    "enrollmentCodeSource","fromNumber","id","initialDeclineReason","leadAddressCity","leadAddressCounty",
    "leadAddressLine1","leadAddressLine2","leadAddressState","leadAddressZip","leadEmail","leadFirstName",
    "leadLastName","leadPhone","queueId","queueName","result","resultCategory","source","sourceCustomName",
    "toNumber","wasBlocked","wasConnected","wasQueueVoicemail","wasVoicemail"
]

# -------------------------
# Helpers
# -------------------------
def parse_date(series: pd.Series) -> pd.Series:
    """Pick the parsing with most valid timestamps; then try common formats."""
    a = pd.to_datetime(series, errors="coerce")
    b = pd.to_datetime(series, errors="coerce", dayfirst=True)
    parsed = b if b.notna().sum() > a.notna().sum() else a
    if parsed.notna().any():
        return parsed
    fmts = ["%d/%m/%Y %I:%M %p", "%m/%d/%Y %I:%M %p", "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S", "%d/%m/%Y %H:%M", "%m/%d/%Y %H:%M"]
    for fmt in fmts:
        try:
            alt = pd.to_datetime(series, errors="coerce", format=fmt)
            if alt.notna().any():
                return alt
        except Exception:
            pass
    return a  # NaT fallback

def to_bool(series: pd.Series) -> pd.Series:
    def _cast(v):
        if pd.isna(v): return np.nan
        if isinstance(v, (bool, np.bool_)): return bool(v)
        s = str(v).strip().lower()
        return s in {"y","yes","true","1","t"}
    try:
        return series.apply(_cast)
    except Exception:
        return pd.Series([np.nan]*len(series), index=series.index)

def to_number(series: pd.Series) -> pd.Series:
    # "10%" -> 10.0 (you can divide by 100 later if needed)
    return pd.to_numeric(series.astype(str).str.replace('%','', regex=False), errors="coerce")

def load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix.lower() in {".xlsx", ".xls"}:
        try:
            df = pd.read_excel(path)
        except Exception:
            df = pd.read_excel(path, engine="openpyxl")
    else:
        # Try common encodings
        df, last = None, None
        for enc in ["utf-8","utf-8-sig","cp1252","latin1"]:
            try:
                df = pd.read_csv(path, encoding=enc)
                break
            except Exception as e:
                last = e
        if df is None:
            raise RuntimeError(f"Failed reading {path.name} with common encodings: {last}")
    df.columns = [c.strip() for c in df.columns]
    return df

def safe_col(df: pd.DataFrame, name: str) -> bool:
    return name in df.columns

# -------------------------
# Domain-specific preprocessing
# -------------------------
def preprocess_complaints(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    # Ensure known columns exist
    for c in DATE_COLS_CANDIDATES + BOOL_COLS_CANDIDATES + NUM_COLS_CANDIDATES + TEXT_COLS:
        if c not in df.columns:
            df[c] = np.nan

    # Dates
    for c in DATE_COLS_CANDIDATES:
        df[c] = parse_date(df[c])

    # Booleans
    for c in BOOL_COLS_CANDIDATES:
        df[c] = to_bool(df[c])

    # Numerics
    for c in NUM_COLS_CANDIDATES:
        df[c] = to_number(df[c])

    # Complaint Source
    df["Complaint Source"] = np.select(
        [df["CTM"].eq(True), df["CTM"].eq(False)], ["CMS (CTM)","Carrier-Derived"], default="(Unknown)"
    )

    # Response date
    if safe_col(df,"Response Submited"):
        df["Response Submited"] = parse_date(df["Response Submited"])
        resp = df["Response Submited"]
    elif safe_col(df,"Response Submitted"):
        df["Response Submitted"] = parse_date(df["Response Submitted"])
        resp = df["Response Submitted"]
    else:
        resp = pd.Series(pd.NaT, index=df.index)

    deadline = df["Deadline Date"] if safe_col(df,"Deadline Date") else pd.Series(pd.NaT, index=df.index)
    occurred = df["Date of Occurence"] if safe_col(df,"Date of Occurence") else pd.Series(pd.NaT, index=df.index)

    # On-time / overdue / days metrics
    on_time = pd.Series(pd.NA, index=df.index, dtype="boolean")
    both = resp.notna() & deadline.notna()
    on_time[both] = resp[both] <= deadline[both]
    df["On Time (<= Deadline)"] = on_time

    now = pd.Timestamp.utcnow().tz_localize(None)
    df["Overdue (No Submit > Deadline)"] = resp.isna() & deadline.notna() & (deadline < now)
    df["Days to Submit"] = (resp - occurred).dt.days

    df["Days Late"] = pd.Series(pd.NA, index=df.index, dtype="Float64")
    late_submits = resp.notna() & deadline.notna() & (resp > deadline)
    df.loc[late_submits, "Days Late"] = (resp - deadline).dt.days
    open_overdue = resp.isna() & deadline.notna() & (deadline < now)
    df.loc[open_overdue, "Days Late"] = (now - deadline).dt.days

    # Normalize text
    for c in TEXT_COLS:
        df[c] = df[c].astype(str).str.strip()

    # Fill unknowns for grouping
    for c in ["Team Name","Agent Name","Carrier Name"]:
        if c in df.columns:
            df[c] = df[c].replace({"nan": np.nan, "None": np.nan}).fillna("(Unknown)")

    return df

def preprocess_enrollments(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    if df.empty:
        return df, "createdAt"
    # Ensure expected columns exist
    for c in ENR_EXPECTED:
        if c not in df.columns:
            df[c] = np.nan

    # Dates
    date_col = "createdAt" if "createdAt" in df.columns else None
    if not date_col:
        # fallback to first column containing "date" or "created" tokens
        cand = [c for c in df.columns if any(t in c.lower() for t in ["date","created","time"])]
        date_col = cand[0] if cand else df.columns[0]
    df[date_col] = parse_date(df[date_col])

    # Booleans
    for c in ["wasBlocked","wasConnected","wasQueueVoicemail","wasVoicemail","billable"]:
        if c in df.columns:
            df[c] = to_bool(df[c])

    # Normalize mapping columns (we’re not using regex here)
    for c in ["result","resultCategory","queueName","source","sourceCustomName",
              "enrollmentCode","enrollmentCodeSource"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    return df, date_col

# -------------------------
# Metric builders
# -------------------------
def monthly_complaints(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["month","complaints_total","ctm_share","on_time_rate",
                                     "median_days_to_submit","avg_days_late"])
    work = df.copy()

    # Exclude unknown carriers if configured
    if EXCLUDE_UNKNOWN_CARRIERS and "Carrier Name" in work.columns:
        work = work[work["Carrier Name"].ne("(Unknown)")]

    # Default timeline dimension = Date of Occurence (as requested)
    base = work["Date of Occurence"] if "Date of Occurence" in work.columns else work["Enrollment Date"]
    work["month"] = base.dt.to_period("M").dt.to_timestamp()

    grp = work.groupby("month", dropna=False)
    out = pd.DataFrame({
        "complaints_total": grp.size()
    })
    # Shares as decimals (0..1) so you can set “Percentage” format in Databox
    out["ctm_share"] = grp["Complaint Source"].apply(
        lambda s: (s == "CMS (CTM)").mean() if len(s) else np.nan
    )
    out["on_time_rate"] = grp["On Time (<= Deadline)"].mean()
    out["median_days_to_submit"] = grp["Days to Submit"].median()
    out["avg_days_late"] = grp["Days Late"].mean()
    out = out.reset_index()
    return out

def monthly_enrollments(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["month","enrollments_total"])
    mask = df[date_col].notna()
    if REQUIRE_CONNECTED and "wasConnected" in df.columns:
        mask &= (df["wasConnected"] == True)
    dd = df.loc[mask].copy()
    if dd.empty:
        return pd.DataFrame(columns=["month","enrollments_total"])
    dd["month"] = dd[date_col].dt.to_period("M").dt.to_timestamp()
    out = dd.groupby("month").size().reset_index(name="enrollments_total")
    return out

def complaints_30d_rate(complaints_df: pd.DataFrame,
                        enrollments_monthly_df: pd.DataFrame) -> pd.DataFrame:
    """Complaints with occurrence within 30 days of *Enrollment Date*, bucketed by Enrollment Month."""
    need_cols = {"Enrollment Date", "Date of Occurence"}
    if complaints_df.empty or not need_cols.issubset(complaints_df.columns) or enrollments_monthly_df.empty:
        return pd.DataFrame(columns=["month","complaints_30d","complaints_30d_rate"])
    c = complaints_df.copy()
    # Dimension filters consistent with other calcs
    if EXCLUDE_UNKNOWN_CARRIERS and "Carrier Name" in c.columns:
        c = c[c["Carrier Name"].ne("(Unknown)")]
    c = c.dropna(subset=["Enrollment Date","Date of Occurence"])
    if c.empty:
        return pd.DataFrame(columns=["month","complaints_30d","complaints_30d_rate"])
    c["days_from_enroll"] = (c["Date of Occurence"] - c["Enrollment Date"]).dt.days
    c = c[(c["days_from_enroll"] >= 0) & (c["days_from_enroll"] <= 30)]
    if c.empty:
        return pd.DataFrame(columns=["month","complaints_30d","complaints_30d_rate"])
    c["month"] = c["Enrollment Date"].dt.to_period("M").dt.to_timestamp()
    m = c.groupby("month").size().reset_index(name="complaints_30d")
    merged = enrollments_monthly_df.merge(m, on="month", how="left")
    merged["complaints_30d"] = merged["complaints_30d"].fillna(0)
    merged["complaints_30d_rate"] = merged.apply(
        lambda r: np.nan if (pd.isna(r["enrollments_total"]) or r["enrollments_total"] == 0)
        else r["complaints_30d"] / r["enrollments_total"], axis=1
    )
    return merged[["month","complaints_30d","complaints_30d_rate"]]

# -------------------------
# Orchestration
# -------------------------
def build_databox_sheet(complaints_path: Path,
                        enrollments_path: Path,
                        out_xlsx: Path,
                        out_csv: Path) -> pd.DataFrame:
    # Load
    raw_c = load_table(complaints_path)
    raw_e = load_table(enrollments_path)

    # Preprocess
    comp = preprocess_complaints(raw_c) if not raw_c.empty else pd.DataFrame()
    enr, date_col = preprocess_enrollments(raw_e) if not raw_e.empty else (pd.DataFrame(), "createdAt")

    # Aggregates
    m_comp = monthly_complaints(comp)
    m_enr = monthly_enrollments(enr, date_col)
    m_join = pd.merge(m_comp, m_enr, on="month", how="outer").sort_values("month")

    # Ratios
    if not m_join.empty:
        m_join["complaints_per_1000_enrollments"] = np.where(
            (m_join["enrollments_total"] > 0) & (~m_join["enrollments_total"].isna()),
            (m_join["complaints_total"] / m_join["enrollments_total"]) * 1000,
            np.nan
        )

    # 30-day cohort rate
    m_30 = complaints_30d_rate(comp, m_enr)
    if not m_30.empty:
        m_join = m_join.merge(m_30, on="month", how="left")

    # Final Databox-ready frame (Date column first, YYYY-MM-DD)
    out = m_join.copy()
    out.insert(0, "Date", out["month"].dt.strftime("%Y-%m-%d"))
    out = out.drop(columns=["month"])

    # Ensure numeric dtypes for Databox (you can set % formatting in-app)
    num_cols = [c for c in out.columns if c != "Date"]
    out[num_cols] = out[num_cols].apply(pd.to_numeric, errors="coerce")

    # Save
    out.to_excel(out_xlsx, sheet_name="databox", index=False)
    out.to_csv(out_csv, index=False)

    return out

def main():
    parser = argparse.ArgumentParser(description="Build a Databox-ready sheet from compliance + enrollments exports.")
    parser.add_argument("--in-dir", type=str, default=".", help="Folder containing compliance2.* and enrollments.*")
    parser.add_argument("--complaints", type=str, default="compliance2.csv", help="Complaints file (csv/xlsx)")
    parser.add_argument("--enrollments", type=str, default="enrollments.csv", help="Enrollments file (csv/xlsx)")
    parser.add_argument("--out-xlsx", type=str, default="databox_feed.xlsx", help="Output Excel path")
    parser.add_argument("--out-csv", type=str, default="databox_feed.csv", help="Output CSV path")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    c_path = in_dir / args.complaints
    e_path = in_dir / args.enrollments
    out_xlsx = in_dir / args.out_xlsx
    out_csv = in_dir / args.out_csv

    # Accept xlsx fallbacks if csv not found (and vice versa)
    def find_file(p: Path) -> Path:
        if p.exists(): return p
        alt = p.with_suffix(".xlsx") if p.suffix.lower() != ".xlsx" else p.with_suffix(".csv")
        return alt if alt.exists() else p

    c_path = find_file(c_path)
    e_path = find_file(e_path)

    df = build_databox_sheet(c_path, e_path, out_xlsx, out_csv)
    print(f"Rows written: {len(df)}")
    print(f"Saved: {out_xlsx} and {out_csv}")

if __name__ == "__main__":
    main()
