import io
import math
from datetime import datetime, timezone
from typing import Optional, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
    # Try standard parsing
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.notna().any():
        return parsed
    # Try day-first parsing (e.g., 23/08/2025 07:44 PM)
    parsed = pd.to_datetime(series, errors="coerce", dayfirst=True)
    if parsed.notna().any():
        return parsed
    # Try specific common formats
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
    return pd.to_datetime(series, errors="coerce")


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
        # CSV: try multiple encodings
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

    # Derived complaint source
    df["Complaint Source"] = np.where(df["CTM"] == True, "CMS (CTM)", "Carrier-Derived")

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

    # On-time logic: submitted <= deadline
    on_time = (response_date.notna()) & (deadline.notna()) & (response_date <= deadline)
    df["On Time (<= Deadline)"] = on_time

    # Overdue logic: not submitted and today > deadline
    now = pd.Timestamp(datetime.now(timezone.utc)).tz_localize(None)
    overdue = (response_date.isna()) & (deadline.notna()) & (deadline < now)
    df["Overdue (No Submit > Deadline)"] = overdue

    # Days to submit (from occurrence)
    df["Days to Submit"] = (response_date - occurred).dt.days

    # Days late
    days_late = pd.Series(0.0, index=df.index)
    late_submits = (response_date.notna()) & (deadline.notna()) & (response_date > deadline)
    days_late[late_submits] = (response_date[late_submits] - deadline[late_submits]).dt.days
    open_overdue = (response_date.isna()) & (deadline.notna()) & (deadline < now)
    days_late[open_overdue] = (now - deadline[open_overdue]).dt.days
    df["Days Late"] = days_late

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
    lowered = {str(v).strip().lower() for v in unique_vals}
    picks = []
    keywords = [
        "sale", "sold", "enroll", "policy", "bind", "conversion",
        "app", "submit", "submitted", "s_app_submitted"
    ]
    for v in unique_vals:
        vl = str(v).strip().lower()
        if any(k in vl for k in keywords):
            picks.append(v)
    # If nothing matched, return empty (let user pick)
    return picks


def pct(a: float) -> str:
    if a is None or np.isnan(a):
        return "—"
    return f"{a*100:.1f}%"


# -------------------------
# Data loading from local directory
# -------------------------
base_dir = Path(__file__).resolve().parent
complaints_path = base_dir / "compliance.csv"
enrollments_path = base_dir / "enrollments.csv"

raw_df = load_csv_from_path(complaints_path)
enr_raw = load_csv_from_path(enrollments_path)

# Handle missing/empty complaints more gracefully
if raw_df.empty:
    st.info(f"Could not load complaints data from {complaints_path}. Place 'compliance.csv' in the app directory.")
    _df = pd.DataFrame(columns=EXPECTED_COLS)
    has_complaints_data = False
else:
    # Preprocess complaints
    _df = preprocess_complaints(raw_df.copy())
    has_complaints_data = not _df.empty

# -------------------------
# Filters (complaints)
# -------------------------
if has_complaints_data:
    st.sidebar.header("Filters")
    all_dims = [c for c in DATE_COLS_CANDIDATES if safe_col(_df, c)]
    if not all_dims:
        all_dims = ["Date of Occurence"]

    try:
        date_dim = st.sidebar.selectbox(
            "Date dimension for complaint trends",
            options=all_dims,
            index=min(1, len(all_dims) - 1)
        )
    except Exception:
        date_dim = all_dims[0]

    _df["_trend_month"] = _df[date_dim].dt.to_period("M").dt.to_timestamp()

    min_d, max_d = _df[date_dim].min(), _df[date_dim].max()
    if pd.isna(min_d) or pd.isna(max_d):
        min_d = datetime(2024, 1, 1)
        max_d = datetime.today()

    start, end = st.sidebar.date_input("Complaints date range", value=(min_d, max_d))
    mask_date = (_df[date_dim] >= pd.to_datetime(start)) & (_df[date_dim] <= pd.to_datetime(end))

    teams = sorted(_df["Team Name"].dropna().unique().tolist()) if safe_col(_df, "Team Name") else []
    agents = sorted(_df["Agent Name"].dropna().unique().tolist()) if safe_col(_df, "Agent Name") else []
    carriers = sorted(_df["Carrier Name"].dropna().unique().tolist()) if safe_col(_df, "Carrier Name") else []

    sel_teams = st.sidebar.multiselect("Team(s)", teams)
    sel_agents = st.sidebar.multiselect("Agent(s)", agents)
    sel_carriers = st.sidebar.multiselect("Carrier(s)", carriers)

    sel_ctm = st.sidebar.multiselect("Complaint Source", ["CMS (CTM)", "Carrier-Derived"])  # empty = all

    mask = mask_date.copy()
    if sel_teams:
        mask &= _df["Team Name"].isin(sel_teams)
    if sel_agents:
        mask &= _df["Agent Name"].isin(sel_agents)
    if sel_carriers:
        mask &= _df["Carrier Name"].isin(sel_carriers)
    if sel_ctm:
        mask &= _df["Complaint Source"].isin(sel_ctm)

    f = _df.loc[mask].copy()

    #st.caption(f"Filtered complaint rows: **{len(f):,}** of **{len(_df):,}** total.")
else:
    # Define placeholders so later sections can safely check
    f = pd.DataFrame()
    start = None
    end = None

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

    require_connected = True

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
        extra_regex = "sale|sold|enroll|policy|bind|app|submitted|s_app_submitted"

        enr_mask = pd.Series(True, index=enr_raw.index)
        enr_mask &= enr_raw[enr_date_col].notna()
        if require_connected and safe_col(enr_raw, "wasConnected"):
            enr_mask &= (enr_raw["wasConnected"] == True)
        if picked_vals:
            enr_mask &= enr_raw[map_col].isin(picked_vals)
        if extra_regex:
            try:
                rx = enr_raw["result"].fillna("").str.contains(extra_regex, case=False, regex=True) if safe_col(enr_raw, "result") else pd.Series(False, index=enr_raw.index)
                rc = enr_raw["resultCategory"].fillna("").str.contains(extra_regex, case=False, regex=True) if safe_col(enr_raw, "resultCategory") else pd.Series(False, index=enr_raw.index)
                enr_mask &= (rx | rc | enr_raw[map_col].astype(str).str.contains(extra_regex, case=False, regex=True))
            except Exception:
                pass

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
        median_days_submit = (f["Days to Submit"].median() if f["Days to Submit"].notna().any() else np.nan)
        st.metric("CTM Share", pct(ctm_share))
        st.metric("Median Days to Submit", "—" if np.isnan(median_days_submit) else f"{median_days_submit:.1f} d")

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
            st.metric("Avg Enrollments/Month", "—" if np.isnan(avg_enr) else f"{avg_enr:.1f}")
        else:
            st.metric("Total Enrollments", "—")
            st.metric("Avg Enrollments/Month", "—")
else:
    # Only show enrollments KPIs if available
    col = st.columns([1])[0]
    with col:
        if not enrollments_monthly.empty:
            em = enrollments_monthly.copy()
            total_enr = int(em["enrollments"].sum()) if not em.empty else 0
            avg_enr = float(em["enrollments"].mean()) if not em.empty else np.nan
            st.metric("Total Enrollments", f"{total_enr:,}")
            st.metric("Avg Enrollments/Month", "—" if np.isnan(avg_enr) else f"{avg_enr:.1f}")

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

    # 2) Date of Occurence trends: weekly and monthly
    if "Date of Occurence" in f.columns and f["Date of Occurence"].notna().any():
        occ = f.dropna(subset=["Date of Occurence"]).copy()
        # Week buckets (start of week)
        occ["_week"] = occ["Date of Occurence"].dt.to_period("W").dt.start_time
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

    # 2b) Enrollment Date duplicate trends: weekly and monthly
    if "Enrollment Date" in f.columns and f["Enrollment Date"].notna().any():
        enr = f.dropna(subset=["Enrollment Date"]).copy()
        enr["_week_enr"] = enr["Enrollment Date"].dt.to_period("W").dt.start_time
        enr["_month_enr"] = enr["Enrollment Date"].dt.to_period("M").dt.to_timestamp()

        weekly_enr = enr.groupby("_week_enr").size().reset_index(name="count")
        monthly_enr = enr.groupby("_month_enr").size().reset_index(name="count")

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
    # compute only when we had complaints charts (vol) available
    vol = (
        f.assign(month=f["_trend_month"])\
         .groupby("month", dropna=False).size().reset_index(name="count")
    )
    if not vol.empty and not enrollments_monthly.empty:
        comp_monthly = vol.groupby("month")["count"].sum().reset_index(name="complaints")
        ratio_df = comp_monthly.merge(enrollments_monthly, on="month", how="inner")
        ratio_df["complaints_per_1000_enrollments"] = (ratio_df["complaints"] / ratio_df["enrollments"]) * 1000

        r_fig = go.Figure()
        r_fig.add_bar(x=ratio_df["month"], y=ratio_df["complaints"], name="Complaints", text=ratio_df["complaints"], textposition="outside")
        r_fig.add_scatter(x=ratio_df["month"], y=ratio_df["enrollments"], name="Enrollments", mode="lines+markers+text", text=ratio_df["enrollments"], textposition="top center", yaxis="y2")
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
        rr_fig.update_traces(mode="lines+markers+text", text=ratio_df["complaints_per_1000_enrollments"].round(1), textposition="top center")
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
# 30-Day Complaint Rate (by Enrollment Month)
# -------------------------
st.subheader("30-Day Complaint Rate")

if has_complaints_data and not enrollments_monthly.empty and "Enrollment Date" in f.columns and "Date of Occurence" in f.columns:
    c = f.dropna(subset=["Enrollment Date", "Date of Occurence"]).copy()
    if not c.empty:
        c["days_from_enroll"] = (c["Date of Occurence"] - c["Enrollment Date"]).dt.days
        c = c[(c["days_from_enroll"] >= 0) & (c["days_from_enroll"] <= 30)]

        if not c.empty:
            c["month"] = c["Enrollment Date"].dt.to_period("M").dt.to_timestamp()
            m = c.groupby("month").size().reset_index(name="complaints_30d")
            rate_df2 = enrollments_monthly.merge(m, on="month", how="left")
            rate_df2["complaints_30d"] = rate_df2["complaints_30d"].fillna(0)

            if start is not None and end is not None:
                start_ts = pd.to_datetime(start).to_period("M").to_timestamp()
                end_ts = pd.to_datetime(end).to_period("M").to_timestamp()
                rate_df2 = rate_df2[(rate_df2["month"] >= start_ts) & (rate_df2["month"] <= end_ts)]

            rate_df2["rate"] = rate_df2["complaints_30d"] / rate_df2["enrollments"]
            rate_df2["rate"] = rate_df2["rate"].replace([np.inf, -np.inf], np.nan)

            if not rate_df2.empty:
                rate_fig = px.line(
                    rate_df2,
                    x="month",
                    y="rate",
                    markers=True,
                    title="30-Day Complaint Rate by Enrollment Cohort = complaints < 30d / enrollments",
                    labels={"month": "Enrollment Month", "rate": "Rate"},
                    custom_data=["complaints_30d", "enrollments"],
                )
                rate_fig.update_yaxes(tickformat=",.0%", autorange=True)
                rate_fig.update_traces(
                    mode="lines+markers+text",
                    text=(rate_df2["rate"] * 100).round(1).astype(str) + "%",
                    textposition="top center",
                    hovertemplate="%{x|%b %Y}<br>%{customdata[0]:,.0f} / %{customdata[1]:,.0f} = %{y:.1%}<extra></extra>",
                )
                st.plotly_chart(rate_fig, use_container_width=True)

                # Companion counts bar chart (enrollments & 30d complaints)
                counts_df = rate_df2[["month", "enrollments", "complaints_30d"]].copy()
                counts_long = counts_df.melt(id_vars="month", var_name="metric", value_name="count")
                label_map = {"enrollments": "Enrollments", "complaints_30d": "Complaints ≤30d"}
                counts_long["metric"] = counts_long["metric"].map(label_map)
                counts_fig = px.bar(
                    counts_long,
                    x="month",
                    y="count",
                    color="metric",
                    barmode="group",
                    title="Monthly Counts: Enrollments vs Complaints < 30d",
                    labels={"month": "Month", "count": "Count", "metric": "Series"},
                    text_auto=True,
                )
                st.plotly_chart(counts_fig, use_container_width=True)
            else:
                st.info("No months available after applying the date range.")
        else:
            st.info("No complaints with occurrence within 30 days of enrollment in current selection.")
    else:
        st.info("No complaints rows with both 'Enrollment Date' and 'Date of Occurence'.")
else:
    st.info("Need complaints with 'Enrollment Date' and 'Date of Occurence', and enrollments data to compute 30-day rate.")

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

    # Exclude unknown carriers from visualizations
    metrics_df = metrics_df[metrics_df["carrier"] != "(Unknown)"]
    metrics_df = metrics_df.sort_values("complaints", ascending=False)
    selection = metrics_df.copy()
    selected_carriers = selection["carrier"].tolist()

    # 1) Composition: 100% stacked (Carrier-Derived vs CTM)
    comp = (
        f[f["Carrier Name"].isin(selected_carriers)]
          .groupby(["Carrier Name", "Complaint Source"], dropna=False)
          .size().reset_index(name="count")
    )
    if not comp.empty:
        comp_fig = px.bar(
            comp, x="Carrier Name", y="count", color="Complaint Source",
            title="Complaint Source Mix by Carrier",
            labels={"Carrier Name": "Carrier", "count": "Share"},
            text_auto=True,
        )
        comp_fig.update_layout(barmode="stack", barnorm="percent", xaxis_title=None, legend_title_text="Source")
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
            # Order carriers by total volume, after excluding unknowns above
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
            else:
                st.info("No carriers available after excluding unknown.")
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
