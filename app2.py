import io
import math
from datetime import datetime, timezone
from typing import Optional, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from pathlib import Path

# -------------------------
# Config & style
# -------------------------
st.set_page_config(
    page_title="Compliance Dashboard | Enrollments vs Complaints",
    page_icon=None,
    layout="wide",
)

st.markdown(
    """
    <style>
    .metric-small .stMetric { padding: 0.25rem 0.5rem; }
    .muted { color: #666; font-size: 0.9rem; }
    .pill { display:inline-block; padding:2px 8px; border-radius:12px; background:#f1f5f9; margin-left:6px; font-size:0.8rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Helpers
# -------------------------
DATE_COLS_CANDIDATES = [
    "Enrollment Date",
    "Date of Occurence",  # spelled as provided
    "Follow-Up Date",
    "Deadline Date",
    "Carrier Ruling Date",
    "Last edited time",
    # Added common variants for complaint received date
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

OPTIONAL_INFO_COLS = [
    "Activity Log",
    "Case Activity",
    "Case Info",
    "Enrollment Info",
    "Remedial Actions",
    "Recording File",
    "Member Phone",
    "Created by",
    "Last edited by",
]

EXPECTED_COLS = (
    DATE_COLS_CANDIDATES + BOOL_COLS_CANDIDATES + NUM_COLS_CANDIDATES + TEXT_COLS + OPTIONAL_INFO_COLS
)

# Enrollments-raw expected columns (from your CSV spec)
ENR_EXPECTED = [
    "agencyId","agencyName","agentEmail","agentFirstName","agentId","agentLastName","agentNpn",
    "billable","billableCostBase","billableCostMinutes","billableCostTotal","billableSeconds","callType",
    "createdAt","direction","duration","durationInCall","durationInQueue","enrollmentCode","enrollmentCodeNote",
    "enrollmentCodeSource","fromNumber","id","initialDeclineReason","leadAddressCity","leadAddressCounty",
    "leadAddressLine1","leadAddressLine2","leadAddressState","leadAddressZip","leadEmail","leadFirstName",
    "leadLastName","leadPhone","queueId","queueName","result","resultCategory","source","sourceCustomName",
    "toNumber","wasBlocked","wasConnected","wasQueueVoicemail","wasVoicemail"
]

# ---------- Casting helpers ----------
def parse_date(series: pd.Series) -> pd.Series:
    """Choose the parsing that yields the most valid timestamps; then fall back to common formats."""
    a = pd.to_datetime(series, errors="coerce")  # month-first
    b = pd.to_datetime(series, errors="coerce", dayfirst=True)  # day-first
    parsed = b if b.notna().sum() > a.notna().sum() else a
    if parsed.notna().any():
        return parsed
    candidate_formats = [
        "%d/%m/%Y %I:%M %p",
        "%m/%d/%Y %I:%M %p",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%m/%d/%Y %H:%M",
    ]
    for fmt in candidate_formats:
        try:
            parsed = pd.to_datetime(series, errors="coerce", format=fmt)
            if parsed.notna().any():
                return parsed
        except Exception:
            continue
    return a  # all NaT

def to_bool(series: pd.Series) -> pd.Series:
    def _cast(v):
        if pd.isna(v):
            return np.nan
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        s = str(v).strip().lower()
        return s in {"y", "yes", "true", "1", "t"}
    try:
        return series.apply(_cast)
    except Exception:
        return pd.Series([np.nan] * len(series), index=series.index)

def to_number(series: pd.Series) -> pd.Series:
    # Note: "10%" -> 10.0 (not 0.10). Divide by 100 later if using as a fraction.
    return pd.to_numeric(series.astype(str).str.replace('%', '', regex=False), errors="coerce")

def safe_col(df: pd.DataFrame, name: str) -> bool:
    return name in df.columns

@st.cache_data(show_spinner=False)
def load_table(uploaded_file, sheet: Optional[str] = None) -> pd.DataFrame:
    if uploaded_file is None:
        return pd.DataFrame()
    name = uploaded_file.name.lower()
    if name.endswith((".xlsx", ".xls")):
        try:
            if sheet:
                df = pd.read_excel(uploaded_file, sheet_name=sheet)
            else:
                df = pd.read_excel(uploaded_file)
        except Exception:
            df = pd.read_excel(uploaded_file, engine="openpyxl")
    else:
        encodings_to_try = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
        last_error = None
        df = None
        for enc in encodings_to_try:
            try:
                try:
                    uploaded_file.seek(0)
                except Exception:
                    pass
                df = pd.read_csv(uploaded_file, encoding=enc)
                break
            except UnicodeDecodeError as e:
                last_error = e
            except Exception as e:
                last_error = e
        if df is None:
            st.error(f"Failed to read CSV with common encodings. Last error: {last_error}")
            return pd.DataFrame()
    df.columns = [c.strip() for c in df.columns]
    return df

@st.cache_data(show_spinner=False)
def load_csv_from_path(file_path) -> pd.DataFrame:
    p = Path(file_path)
    if not p.exists():
        return pd.DataFrame()
    encodings_to_try = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    last_error = None
    for enc in encodings_to_try:
        try:
            df = pd.read_csv(p, encoding=enc)
            df.columns = [c.strip() for c in df.columns]
            return df
        except Exception as e:
            last_error = e
            continue
    st.error(f"Failed to read {p.name} with common encodings. Last error: {last_error}")
    return pd.DataFrame()

# ---------- Complaints preprocessing ----------
def preprocess_complaints(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    # Ensure expected columns exist (create empty if missing)
    for col in EXPECTED_COLS:
        if col not in df.columns:
            df[col] = np.nan

    # Dates
    for col in DATE_COLS_CANDIDATES:
        if safe_col(df, col):
            df[col] = parse_date(df[col])

    # Booleans
    for col in BOOL_COLS_CANDIDATES:
        if safe_col(df, col):
            df[col] = to_bool(df[col])

    # Numerics
    for col in NUM_COLS_CANDIDATES:
        if safe_col(df, col):
            df[col] = to_number(df[col])

    # Derived complaint source (do not coerce NaNs into Carrier-Derived)
    df["Complaint Source"] = np.select(
        [df["CTM"].eq(True), df["CTM"].eq(False)],
        ["CMS (CTM)", "Carrier-Derived"],
        default="(Unknown)"
    )

    # Response dates — prefer "Response Submited" as provided
    if safe_col(df, "Response Submited"):
        df["Response Submited"] = parse_date(df["Response Submited"])  # keep original name
        response_date = df["Response Submited"]
    elif safe_col(df, "Response Submitted"):
        df["Response Submitted"] = parse_date(df["Response Submitted"])  # fallback if spelling fixed
        response_date = df["Response Submitted"]
    else:
        response_date = pd.Series(pd.NaT, index=df.index)

    # SLA / Deadlines
    deadline = df["Deadline Date"] if safe_col(df, "Deadline Date") else pd.Series(pd.NaT, index=df.index)
    occurred = df["Date of Occurence"] if safe_col(df, "Date of Occurence") else pd.Series(pd.NaT, index=df.index)

    # On-time logic: nullable boolean; unknown stays NA
    now = pd.Timestamp.now(tz="Asia/Beirut").tz_localize(None)
    on_time = pd.Series(pd.NA, index=df.index, dtype="boolean")
    mask_both = response_date.notna() & deadline.notna()
    on_time[mask_both] = response_date[mask_both] <= deadline[mask_both]
    df["On Time (<= Deadline)"] = on_time

    # Overdue logic: not submitted and today > deadline
    overdue = (response_date.isna()) & (deadline.notna()) & (deadline < now)
    df["Overdue (No Submit > Deadline)"] = overdue

    # Days to submit (from occurrence)
    df["Days to Submit"] = (response_date - occurred).dt.days

    # Days late: only when late or open-overdue; else NaN
    df["Days Late"] = pd.Series(pd.NA, index=df.index, dtype="Float64")
    late_submits = (response_date.notna()) & (deadline.notna()) & (response_date > deadline)
    df.loc[late_submits, "Days Late"] = (response_date - deadline).dt.days
    open_overdue = (response_date.isna()) & (deadline.notna()) & (deadline < now)
    df.loc[open_overdue, "Days Late"] = (now - deadline).dt.days

    # Month bucket helper (default to Date of Occurence, overridable from UI)
    default_dim = "Date of Occurence" if df["Date of Occurence"].notna().any() else "Enrollment Date"
    base = df[default_dim]
    df["_month"] = base.dt.to_period("M").dt.to_timestamp()

    # Normalize text columns (strip)
    for col in TEXT_COLS:
        if safe_col(df, col):
            df[col] = df[col].astype(str).str.strip()

    # Coerce team/agent empties to "(Unknown)" for grouping
    for col in ["Team Name", "Agent Name", "Carrier Name"]:
        if safe_col(df, col):
            df[col] = df[col].fillna("(Unknown)").replace({"nan": "(Unknown)", "None": "(Unknown)"})

    return df

# ---------- Enrollments preprocessing ----------
def preprocess_enrollments(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    # Add missing columns as empty to avoid KeyErrors later
    for col in ENR_EXPECTED:
        if col not in df.columns:
            df[col] = np.nan
    # Dates & booleans
    for c in ["createdAt"]:
        if safe_col(df, c):
            df[c] = parse_date(df[c])
    for c in ["wasBlocked","wasConnected","wasQueueVoicemail","wasVoicemail","billable"]:
        if safe_col(df, c):
            df[c] = to_bool(df[c])
    # Normalize text fields used for mapping
    for c in ["result","resultCategory","queueName","source","sourceCustomName","enrollmentCode","enrollmentCodeSource"]:
        if safe_col(df, c):
            df[c] = df[c].astype(str).str.strip()
    return df

def default_enrollment_values(unique_vals: List[str]) -> List[str]:
    # Heuristic: preselect values that look like sales/enrollments
    picks = []
    keywords = [
        "sale", "sold", "enroll", "policy", "bind", "conversion",
        "app", "submit", "submitted", "s_app_submitted"
    ]
    for v in unique_vals:
        vl = str(v).strip().lower()
        if any(k in vl for k in keywords):
            picks.append(v)
    return picks

def pct(a: float) -> str:
    if a is None or (isinstance(a, float) and np.isnan(a)):
        return "—"
    return f"{a*100:.1f}%"

# -------------------------
# Data loading from local directory
# -------------------------
base_dir = Path(__file__).resolve().parent
complaints_path = base_dir / "compliance2.csv"
enrollments_path = base_dir / "enrollments2.csv"

raw_df = load_csv_from_path(complaints_path)
enr_raw = load_csv_from_path(enrollments_path)

# -------------------------
# Sidebar controls
# -------------------------
with st.sidebar:
    st.header("Filters")
    # Always checked on page load
    require_connected = st.checkbox(
        "Require connected calls for enrollments",
        value=True,
        help="If checked, only rows with wasConnected == True are counted as enrollments."
    )
    # REMOVED: enrollment keyword regex UI

# -------------------------
# Complaints filters
# -------------------------
if raw_df.empty:
    st.info(f"Could not load complaints data from {complaints_path}. Place 'compliance.csv' in the app directory.")
    _df = pd.DataFrame(columns=EXPECTED_COLS)
    has_complaints_data = False
else:
    _df = preprocess_complaints(raw_df.copy())
    has_complaints_data = not _df.empty

if has_complaints_data:
    all_dims = [c for c in DATE_COLS_CANDIDATES if safe_col(_df, c)] or ["Date of Occurence"]
    # Default to "Date of Occurence" if present
    default_idx = all_dims.index("Date of Occurence") if "Date of Occurence" in all_dims else 0
    try:
        date_dim = st.sidebar.selectbox(
            "Date dimension for complaint trends",
            options=all_dims,
            index=default_idx
        )
    except Exception:
        date_dim = all_dims[default_idx]

    _df["_trend_month"] = _df[date_dim].dt.to_period("M").dt.to_timestamp()

    min_d, max_d = _df[date_dim].min(), _df[date_dim].max()
    if pd.isna(min_d) or pd.isna(max_d):
        min_d = datetime(2024, 1, 1)
        max_d = datetime.today()

    # Default start date set to 01/12/2024 (day-first: 1 December 2024)
    default_start = pd.to_datetime(datetime(2024, 12, 1))
    default_end = max_d if (pd.notna(max_d) and max_d >= default_start) else default_start

    start, end = st.sidebar.date_input("Complaints date range", value=(default_start, default_end))
    mask_date = (_df[date_dim] >= pd.to_datetime(start)) & (_df[date_dim] <= pd.to_datetime(end))

    teams = sorted(_df["Team Name"].dropna().unique().tolist()) if safe_col(_df, "Team Name") else []
    agents = sorted(_df["Agent Name"].dropna().unique().tolist()) if safe_col(_df, "Agent Name") else []
    carriers = sorted(_df["Carrier Name"].dropna().unique().tolist()) if safe_col(_df, "Carrier Name") else []

    sel_teams = st.sidebar.multiselect("Team(s)", teams)
    sel_agents = st.sidebar.multiselect("Agent(s)", agents)
    sel_carriers = st.sidebar.multiselect("Carrier(s)", carriers)

    sel_ctm = st.sidebar.multiselect("Complaint Source", ["CMS (CTM)", "Carrier-Derived", "(Unknown)"])  # empty = all

    mask = mask_date.copy()
    if sel_teams:
        mask &= _df["Team Name"].isin(sel_teams)
    if sel_agents:
        mask &= _df["Agent Name"].isin(sel_agents)
    if sel_carriers:
        mask &= _df["Carrier Name"].isin(sel_carriers)
    if sel_ctm:
        mask &= _df["Complaint Source"].isin(sel_ctm)

    # Exclude unknown carriers from ALL calculations
    if safe_col(_df, "Carrier Name"):
        mask &= _df["Carrier Name"].ne("(Unknown)")

    f = _df.loc[mask].copy()
else:
    f = pd.DataFrame()
    start = None
    end = None
    sel_teams = sel_agents = sel_carriers = sel_ctm = []

# -------------------------
# Enrollments mapping and monthly aggregation
# -------------------------
enrollments_monthly = pd.DataFrame(columns=["month","enrollments"])  # default empty
enr_filtered = pd.DataFrame()

if not enr_raw.empty:
    enr_raw = preprocess_enrollments(enr_raw)

    # Auto-select date column
    candidate_date_cols = [c for c in enr_raw.columns if (
        c.lower() == "createdat" or
        "date" in c.lower() or
        "time" in c.lower() or
        "created" in c.lower()
    )]
    enr_date_cols = [c for c in ["createdAt"] if safe_col(enr_raw, c)] or candidate_date_cols or [enr_raw.columns[0]]
    enr_date_col = enr_date_cols[0]
    if safe_col(enr_raw, enr_date_col):
        enr_raw[enr_date_col] = parse_date(enr_raw[enr_date_col])

    # Auto-detect mapping column and values
    mapping_candidates = [
        "resultCategory","result","enrollmentCode","enrollmentCodeSource",
        "status","disposition","outcome","code"
    ]
    mapping_cols = [c for c in mapping_candidates if safe_col(enr_raw, c)]

    if not mapping_cols:
        # Count all rows per month (optionally require wasConnected)
        enr_mask = pd.Series(True, index=enr_raw.index)
        enr_mask &= enr_raw[enr_date_col].notna()
        if require_connected and safe_col(enr_raw, "wasConnected"):
            enr_mask &= (enr_raw["wasConnected"] == True)
        enr = enr_raw.loc[enr_mask].copy()
        if not enr.empty:
            enr["month"] = enr[enr_date_col].dt.to_period("M").dt.to_timestamp()
            enrollments_monthly = enr.groupby("month").size().reset_index(name="enrollments")
        enr_filtered = enr
    else:
        map_col = mapping_cols[0]
        uniques = sorted([v for v in enr_raw[map_col].dropna().unique().tolist() if str(v).strip() != ""])
        picked_vals = default_enrollment_values(uniques)

        enr_mask = pd.Series(True, index=enr_raw.index)
        enr_mask &= enr_raw[enr_date_col].notna()
        if require_connected and safe_col(enr_raw, "wasConnected"):
            enr_mask &= (enr_raw["wasConnected"] == True)
        if picked_vals:
            enr_mask &= enr_raw[map_col].isin(picked_vals)

        # REMOVED: regex enrichment across result/resultCategory/map_col

        enr = enr_raw.loc[enr_mask].copy()
        if not enr.empty:
            enr["month"] = enr[enr_date_col].dt.to_period("M").dt.to_timestamp()
            enrollments_monthly = enr.groupby("month").size().reset_index(name="enrollments")
        enr_filtered = enr

# -------------------------
# KPIs (complaints-focused)
# -------------------------
if has_complaints_data:
    left, mid, enr_col = st.columns([1,1,1])

    with left:
        total_cases = len(f)
        st.metric("Total Cases", f"{total_cases:,}")

    with mid:
        ctm_share = (f["Complaint Source"].eq("CMS (CTM)").mean() if len(f) else np.nan)
        md_mask = (f["Days to Submit"].notna()) & (f["Days to Submit"] >= 0)
        median_days_submit = f.loc[md_mask, "Days to Submit"].median() if md_mask.any() else np.nan
        st.metric("CTM Share", pct(ctm_share))
        st.metric("Median Days to Submit", "—" if (isinstance(median_days_submit, float) and np.isnan(median_days_submit)) else f"{median_days_submit:.1f} d")

    with enr_col:
        if not enrollments_monthly.empty:
            if start is not None and end is not None:
                start_ts = pd.to_datetime(start)
                end_ts = pd.to_datetime(end)
                em = enrollments_monthly[(enrollments_monthly["month"] >= start_ts) & (enrollments_monthly["month"] <= end_ts)]
            else:
                em = enrollments_monthly.copy()
            total_enr = int(em["enrollments"].sum()) if not em.empty else 0
            avg_enr = float(em["enrollments"].mean()) if not em.empty else np.nan
            st.metric("Total Enrollments", f"{total_enr:,}")
            st.metric("Avg Enrollments/Month", "—" if (isinstance(avg_enr, float) and np.isnan(avg_enr)) else f"{avg_enr:.1f}")
        else:
            st.metric("Total Enrollments", "—")
            st.metric("Avg Enrollments/Month", "—")
else:
    col = st.columns([1])[0]
    with col:
        if not enrollments_monthly.empty:
            em = enrollments_monthly.copy()
            total_enr = int(em["enrollments"].sum()) if not em.empty else 0
            avg_enr = float(em["enrollments"].mean()) if not em.empty else np.nan
            st.metric("Total Enrollments", f"{total_enr:,}")
            st.metric("Avg Enrollments/Month", "—" if (isinstance(avg_enr, float) and np.isnan(avg_enr)) else f"{avg_enr:.1f}")

# -------------------------
# Charts (complaints)
# -------------------------
st.subheader("Trends & Performance (Complaints)")
if has_complaints_data:
    # 1) Volume over time (stacked by source)
    vol = (
        f.assign(month=f["_trend_month"])\
         .groupby(["month", "Complaint Source"], dropna=False).size().reset_index(name="count")
    )
    if not vol.empty:
        vol_fig = px.bar(
            vol, x="month", y="count", color="Complaint Source",
            title="Complaints by Month (Stacked by Source)",
            labels={"month": "Month", "count": "Complaints"},
            text_auto=True,
        )
        vol_fig.update_layout(barmode="stack", legend_title_text="Source", xaxis_title=None)
        st.plotly_chart(vol_fig, use_container_width=True)
    else:
        st.info("No rows for the selected filters to plot monthly volume.")

    # 2) Date of Occurence trends: weekly and monthly (weeks start Monday)
    if "Date of Occurence" in f.columns and f["Date of Occurence"].notna().any():
        occ = f.dropna(subset=["Date of Occurence"]).copy()
        occ["_week"] = occ["Date of Occurence"].dt.to_period("W-MON").dt.start_time
        occ["_month_occ"] = occ["Date of Occurence"].dt.to_period("M").dt.to_timestamp()

        weekly = occ.groupby("_week").size().reset_index(name="count")
        monthly_occ = occ.groupby("_month_occ").size().reset_index(name="count")

        c1, c2 = st.columns(2)
        with c1:
            if not weekly.empty:
                w_fig = px.line(
                    weekly,
                    x="_week",
                    y="count",
                    markers=True,
                    title="Complaints by Week (Date of Occurence)",
                    labels={"_week": "Week", "count": "Complaints"},
                )
                w_fig.update_layout(xaxis_title="Week", yaxis_title="Complaints", showlegend=False)
                w_fig.update_traces(mode="lines+markers+text", text=weekly["count"], textposition="top center")
                st.plotly_chart(w_fig, use_container_width=True)
            else:
                st.info("No Date of Occurence values in range to plot weekly.")
        with c2:
            if not monthly_occ.empty:
                m_fig = px.bar(
                    monthly_occ,
                    x="_month_occ",
                    y="count",
                    title="Complaints by Month (Date of Occurence)",
                    text_auto=True,
                    labels={"_month_occ": "Month", "count": "Complaints"},
                )
                m_fig.update_layout(xaxis_title="Month", yaxis_title="Complaints", showlegend=False)
                st.plotly_chart(m_fig, use_container_width=True)
            else:
                st.info("No Date of Occurence values in range to plot monthly.")
    else:
        st.info("No 'Date of Occurence' column found to compute weekly/monthly trends.")

    # 2b) Enrollment Date duplicate trends: weekly and monthly (weeks start Monday)
    if "Enrollment Date" in f.columns and f["Enrollment Date"].notna().any():
        enr_c = f.dropna(subset=["Enrollment Date"]).copy()
        enr_c["_week_enr"] = enr_c["Enrollment Date"].dt.to_period("W-MON").dt.start_time
        enr_c["_month_enr"] = enr_c["Enrollment Date"].dt.to_period("M").dt.to_timestamp()

        weekly_enr = enr_c.groupby("_week_enr").size().reset_index(name="count")
        monthly_enr = enr_c.groupby("_month_enr").size().reset_index(name="count")

        e1, e2 = st.columns(2)
        with e1:
            if not weekly_enr.empty:
                we_fig = px.line(
                    weekly_enr,
                    x="_week_enr",
                    y="count",
                    markers=True,
                    title="Complaints by Week (Enrollment Date)",
                    labels={"_week_enr": "Week", "count": "Complaints"},
                )
                we_fig.update_layout(xaxis_title="Week", yaxis_title="Complaints", showlegend=False)
                we_fig.update_traces(mode="lines+markers+text", text=weekly_enr["count"], textposition="top center")
                st.plotly_chart(we_fig, use_container_width=True)
            else:
                st.info("No Enrollment Date values in range to plot weekly.")
        with e2:
            if not monthly_enr.empty:
                me_fig = px.bar(
                    monthly_enr,
                    x="_month_enr",
                    y="count",
                    title="Complaints by Month (Enrollment Date)",
                    text_auto=True,
                    labels={"_month_enr": "Month", "count": "Complaints"},
                )
                me_fig.update_layout(xaxis_title="Month", yaxis_title="Complaints", showlegend=False)
                st.plotly_chart(me_fig, use_container_width=True)
            else:
                st.info("No Enrollment Date values in range to plot monthly.")
    else:
        st.info("No 'Enrollment Date' column found to compute weekly/monthly trends.")

# -------------------------
# Complaints vs Enrollments (Dual-axis) + Ratio
# -------------------------
st.subheader("Complaints vs Enrollments + Ratios")

ratio_df = pd.DataFrame()
if has_complaints_data:
    vol = (
        f.assign(month=f["_trend_month"])\
         .groupby("month", dropna=False).size().reset_index(name="count")
    )
    if not vol.empty and not enrollments_monthly.empty:
        comp_monthly = vol.groupby("month")["count"].sum().reset_index(name="complaints")
        ratio_df = comp_monthly.merge(enrollments_monthly, on="month", how="inner")
        ratio_df["complaints_per_1000_enrollments"] = np.where(
            ratio_df["enrollments"] > 0,
            (ratio_df["complaints"] / ratio_df["enrollments"]) * 1000,
            np.nan
        )

        r_fig = go.Figure()
        r_fig.add_bar(x=ratio_df["month"], y=ratio_df["complaints"], name="Complaints",
                      text=ratio_df["complaints"], textposition="outside")
        r_fig.add_scatter(x=ratio_df["month"], y=ratio_df["enrollments"], name="Enrollments",
                          mode="lines+markers+text", text=ratio_df["enrollments"],
                          textposition="top center", yaxis="y2")
        r_fig.update_layout(
            title="Complaints vs Enrollments",
            yaxis=dict(title="Complaints"),
            yaxis2=dict(title="Enrollments", overlaying="y", side="right"),
            xaxis_title=None,
            legend_title_text="Series",
        )
        st.plotly_chart(r_fig, use_container_width=True)

        rr_fig = px.line(
            ratio_df,
            x="month",
            y="complaints_per_1000_enrollments",
            markers=True,
            title="Complaints per 1,000 Enrollments",
            labels={"month": "Month", "complaints_per_1000_enrollments": "Complaints per 1,000 Enrollments"},
        )
        rr_fig.update_layout(showlegend=False)
        rr_fig.update_traces(
            mode="lines+markers+text",
            text=ratio_df["complaints_per_1000_enrollments"].round(1),
            textposition="top center"
        )
        st.plotly_chart(rr_fig, use_container_width=True)
    else:
        if enrollments_monthly.empty:
            st.info("Could not load enrollments from enrollments.csv. Place it in the app directory to compute ratios.")
        else:
            st.info("Not enough complaints data after filters to compute monthly totals.")
else:
    if not enrollments_monthly.empty:
        st.info("Could not load complaints from compliance.csv. Place it in the app directory to compute ratios.")

# -------------------------
# CTM per 1,000 Enrollments
# -------------------------
st.subheader("CTM per 1,000 Enrollments")
if has_complaints_data and not enrollments_monthly.empty:
    ctm_only = f[f["Complaint Source"].eq("CMS (CTM)")].copy()
    if not ctm_only.empty:
        ctm_only["month"] = ctm_only["_trend_month"]
        ctm_monthly = ctm_only.groupby("month").size().reset_index(name="ctm_complaints")

        ctm_ratio = ctm_monthly.merge(enrollments_monthly, on="month", how="inner")
        if not ctm_ratio.empty:
            ctm_ratio["ctm_per_1000_enrollments"] = np.where(
                ctm_ratio["enrollments"] > 0,
                (ctm_ratio["ctm_complaints"] / ctm_ratio["enrollments"]) * 1000,
                np.nan
            )

            ctm_fig = px.line(
                ctm_ratio,
                x="month",
                y="ctm_per_1000_enrollments",
                markers=True,
                title="CTM per 1,000 Enrollments",
                labels={"month": "Month", "ctm_per_1000_enrollments": "CTM per 1,000 Enrollments"},
            )
            ctm_fig.update_layout(showlegend=False)
            ctm_fig.update_traces(
                mode="lines+markers+text",
                text=ctm_ratio["ctm_per_1000_enrollments"].round(1),
                textposition="top center",
            )
            st.plotly_chart(ctm_fig, use_container_width=True)
        else:
            st.info("No overlapping months between CTM complaints and enrollments to compute CTM per 1,000.")
    else:
        st.info("No CTM complaints in the current selection.")
else:
    if not has_complaints_data:
        st.info("Could not load complaints from compliance.csv to compute CTM per 1,000.")
    else:
        st.info("Could not load enrollments from enrollments.csv to compute CTM per 1,000.")

# -------------------------
# 30-Day Complaint Rate (by Enrollment Month)
# -------------------------
st.subheader("Complaint Rate")

# Use total complaints per selected month (aligned with other charts) and compute rate vs enrollments
if has_complaints_data and not f.empty and not enrollments_monthly.empty and "_trend_month" in f.columns:
    comp_monthly_all = (
        f.assign(month=f["_trend_month"])\
         .groupby("month", dropna=False).size().reset_index(name="complaints")
    )

    # Filter months to selected date range for consistency
    if start is not None and end is not None:
        start_ts = pd.to_datetime(start).to_period("M").to_timestamp()
        end_ts = pd.to_datetime(end).to_period("M").to_timestamp()
        comp_monthly_all = comp_monthly_all[(comp_monthly_all["month"] >= start_ts) & (comp_monthly_all["month"] <= end_ts)]

    # Join with enrollments on month
    rate_df2 = comp_monthly_all.merge(enrollments_monthly, on="month", how="left")
    rate_df2["enrollments"] = rate_df2["enrollments"].fillna(0)
    rate_df2["rate"] = rate_df2.apply(
        lambda r: np.nan if (pd.isna(r["enrollments"]) or r["enrollments"] == 0) else r["complaints"] / r["enrollments"],
        axis=1
    )

    if not rate_df2.empty:
        combined_fig = make_subplots(specs=[[{"secondary_y": True}]])
        combined_fig.add_trace(
            go.Bar(
                x=rate_df2["month"],
                y=rate_df2["enrollments"],
                name="Enrollments",
                text=rate_df2["enrollments"],
                textposition="outside",
            ),
            secondary_y=False,
        )
        combined_fig.add_trace(
            go.Bar(
                x=rate_df2["month"],
                y=rate_df2["complaints"],
                name="Complaints",
                text=rate_df2["complaints"],
                textposition="outside",
            ),
            secondary_y=False,
        )
        combined_fig.add_trace(
            go.Scatter(
                x=rate_df2["month"],
                y=rate_df2["rate"],
                name="Complaint Rate",
                mode="lines+markers+text",
                text=(rate_df2["rate"] * 100).round(1).astype(str) + "%",
                textposition="top center",
            ),
            secondary_y=True,
        )
        combined_fig.update_layout(
            title="Complaint Rate by Month (Total Complaints)",
            barmode="group",
            xaxis_title="Month",
            legend_title_text="Series",
        )
        combined_fig.update_yaxes(title_text="Count", secondary_y=False)
        combined_fig.update_yaxes(title_text="Rate", tickformat=",.0%", secondary_y=True)
        st.plotly_chart(combined_fig, use_container_width=True)
    else:
        st.info("No months available after applying the date range.")
else:
    if not has_complaints_data or f.empty:
        st.info("Not enough complaints data after filters to compute monthly totals.")
    elif enrollments_monthly.empty:
        st.info("Could not load enrollments from enrollments.csv. Place it in the app directory to compute rates.")

# -------------------------
# Carrier Comparison — % of Carrier-Derived, SLAs, and Trends
# -------------------------
st.subheader("Carrier Comparison")
if has_complaints_data and not f.empty and safe_col(f, "Carrier Name"):
    by_car = f.groupby("Carrier Name", dropna=False)
    metrics_df = pd.DataFrame({
        "complaints": by_car.size(),
        "carrier_derived_share": by_car["Complaint Source"].apply(lambda s: (s == "Carrier-Derived").mean() if len(s) else np.nan),
        "ctm_share": by_car["Complaint Source"].apply(lambda s: (s == "CMS (CTM)").mean() if len(s) else np.nan),
        "on_time_rate": by_car["On Time (<= Deadline)"].mean(),
        "median_days_submit": by_car["Days to Submit"].median(),
        "avg_days_late": by_car["Days Late"].mean(),
    }).reset_index().rename(columns={"Carrier Name": "carrier"})

    # Already excluded "(Unknown)" carriers from f, so nothing to drop here
    metrics_df = metrics_df.sort_values("complaints", ascending=False)
    selection = metrics_df.copy()
    selected_carriers = selection["carrier"].tolist()

    # 1) Composition: 100% stacked (Carrier-Derived vs CTM) with percent text
    comp = (
        f[f["Carrier Name"].isin(selected_carriers)]
          .groupby(["Carrier Name", "Complaint Source"], dropna=False)
          .size().reset_index(name="count")
    )
    if not comp.empty:
        comp["share"] = comp.groupby("Carrier Name")["count"].transform(lambda s: s / s.sum())
        comp_fig = px.bar(
            comp, x="Carrier Name", y="share", color="Complaint Source",
            title="Complaint Source Mix by Carrier",
            labels={"Carrier Name": "Carrier", "share": "Share"},
            text=comp["share"].mul(100).round(0).astype(int).astype(str) + "%",
        )
        comp_fig.update_layout(barmode="stack", xaxis_title=None, legend_title_text="Source")
        comp_fig.update_yaxes(tickformat=",.0%")
        st.plotly_chart(comp_fig, use_container_width=True)

    # 2) Bubble: On-Time vs % Carrier-Derived, bubble size = volume, color = Avg Days Late
    if not selection.empty:
        bub_fig = px.scatter(
            selection,
            x="on_time_rate",
            y="carrier_derived_share",
            size="complaints",
            color="avg_days_late",
            color_continuous_scale="RdYlGn_r",
            hover_name="carrier",
            size_max=40,
            labels={
                "on_time_rate": "On-Time Rate",
                "carrier_derived_share": "% Carrier-Derived",
                "avg_days_late": "Avg Days Late",
                "complaints": "Complaints",
            },
            title="Carrier Performance Bubble: On-Time vs % Carrier-Derived",
        )
        bub_fig.update_xaxes(tickformat=",.0%", autorange=True)
        bub_fig.update_yaxes(tickformat=",.0%", autorange=True)
        bub_fig.update_layout(xaxis_autorange=True, yaxis_autorange=True)
        st.plotly_chart(bub_fig, use_container_width=True)

    # 4) Heatmap: Monthly complaint volume (limited to selected carriers)
    if "_trend_month" in f.columns and not f.empty and selected_carriers:
        month_counts = (
            f[f["Carrier Name"].isin(selected_carriers)]
             .assign(month=f["_trend_month"]) 
             .groupby(["Carrier Name", "month"], dropna=False).size().reset_index(name="count")
        )
        if not month_counts.empty:
            pivot = month_counts.pivot(index="Carrier Name", columns="month", values="count").fillna(0)
            pivot = pivot.loc[selection.set_index("carrier").index.intersection(pivot.index)]
            if not pivot.empty:
                hm_fig = px.imshow(
                    pivot,
                    aspect="auto",
                    color_continuous_scale="Blues",
                    text_auto=".0f",
                    labels=dict(x="Month", y="Carrier", color="Complaints"),
                    title="Monthly Complaint Volume by Carrier",
                )
                st.plotly_chart(hm_fig, use_container_width=True)

    # -------------------------
    # Agent Leaderboard
    # -------------------------
    st.subheader("Agent Leaderboard (Complaints)")
    if has_complaints_data and "Agent Name" in f.columns and not f.empty:
        grp = f.groupby("Agent Name", dropna=False)
        ctm_cases = grp["Complaint Source"].apply(lambda s: (s == "CMS (CTM)").sum() if len(s) else 0)
        carrier_cases = grp["Complaint Source"].apply(lambda s: (s == "Carrier-Derived").sum() if len(s) else 0)
        leaderboard = pd.DataFrame({
            "Cases": grp.size(),
            "CTM Cases": ctm_cases,
            "Carrier Cases": carrier_cases,
            "On-Time Rate": grp["On Time (<= Deadline)"].mean(),
            "Avg Days to Submit": grp["Days to Submit"].mean(),
            "Avg Days Late": grp["Days Late"].mean(),
            "CTM Share": grp["Complaint Source"].apply(lambda s: (s == "CMS (CTM)").mean() if len(s) else np.nan),
        }).reset_index().sort_values(["Cases", "On-Time Rate"], ascending=[False, False])

        st.dataframe(leaderboard, use_container_width=True)

        csv_bytes = leaderboard.to_csv(index=False).encode("utf-8")
        st.download_button("Download Leaderboard (CSV)", data=csv_bytes, file_name="leaderboard.csv", mime="text/csv")
    else:
        st.info("Upload your complaints file to see the leaderboard.")
else:
    st.info("No carrier data available in the current selection.")

# -------------------------
# Raw/Filtered Data + Export
# -------------------------
st.subheader("Filtered Complaint Rows")
if has_complaints_data:
    st.dataframe(f, use_container_width=True, height=420)

    raw_csv = f.to_csv(index=False).encode("utf-8")
    st.download_button("Download Filtered Data (CSV)", data=raw_csv, file_name="filtered_complaints_rows.csv", mime="text/csv")
