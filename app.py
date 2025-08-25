import io
import math
from datetime import datetime, timezone
from typing import Optional, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# -------------------------
# Config & style
# -------------------------
st.set_page_config(
    page_title="Compliance Dashboard — Enrollments vs Complaints",
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
# Sidebar — uploads & filters
# -------------------------
st.sidebar.header("1) Data Upload")
complaints_file = st.sidebar.file_uploader(
    "Upload Compliance/Complaints file (CSV/XLSX)", type=["csv", "xlsx", "xls"], key="complaints"
)

sheet_name = None
if complaints_file is not None and complaints_file.name.lower().endswith((".xlsx", ".xls")):
    try:
        with pd.ExcelFile(complaints_file) as xl:
            sheets = xl.sheet_names
        if len(sheets) > 1:
            sheet_name = st.sidebar.selectbox("Select sheet", sheets)
    except Exception:
        sheet_name = None

raw_df = load_table(complaints_file, sheet=sheet_name)

st.sidebar.header("2) Enrollments (raw CSV)")
enrollments_raw_file = st.sidebar.file_uploader(
    "Upload Enrollments raw CSV (with columns like createdAt, result, resultCategory, wasConnected, etc.)",
    type=["csv","xlsx","xls"], key="enr_raw",
)

# Handle missing/empty complaints more gracefully
if raw_df.empty:
    st.info("Upload your Compliance/Complaints sheet to begin. Accepted formats: CSV, XLSX.")
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
    st.sidebar.header("3) Filters")
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

    st.caption(f"Filtered complaint rows: **{len(f):,}** of **{len(_df):,}** total.")
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

if enrollments_raw_file is not None:
    enr_raw = load_table(enrollments_raw_file)
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
    left, mid, right, enr_col = st.columns([1,1,1,1])

    with left:
        total_cases = len(f)
        on_time_rate = f["On Time (<= Deadline)"].mean() if len(f) else np.nan
        overdue_cases = f["Overdue (No Submit > Deadline)"].sum() if len(f) else 0
        st.metric("Total Cases", f"{total_cases:,}")
        st.metric("On-Time Rate", pct(on_time_rate))
        st.metric("Overdue Open Cases", f"{int(overdue_cases):,}")

    with mid:
        ctm_share = (f["Complaint Source"].eq("CMS (CTM)").mean() if len(f) else np.nan)
        median_days_submit = (f["Days to Submit"].median() if f["Days to Submit"].notna().any() else np.nan)
        avg_days_late = (f["Days Late"].mean() if f["Days Late"].notna().any() else np.nan)
        st.metric("CTM Share", pct(ctm_share))
        st.metric("Median Days to Submit", "—" if np.isnan(median_days_submit) else f"{median_days_submit:.1f} d")
        st.metric("Avg Days Late", "—" if np.isnan(avg_days_late) else f"{avg_days_late:.1f} d")

    with right:
        remedial_rate = (to_bool(f.get("Remedial Signed", pd.Series())).mean() if "Remedial Signed" in f.columns else np.nan)
        training_rate = (to_bool(f.get("Course Completed", pd.Series())).mean() if "Course Completed" in f.columns else np.nan)
        avg_rec_score = (f["Recording Score"].mean() if "Recording Score" in f.columns else np.nan)
        st.metric("Remedial Signed Rate", pct(remedial_rate))
        st.metric("Training Completion Rate", pct(training_rate))
        st.metric("Avg Recording Score", "—" if np.isnan(avg_rec_score) else f"{avg_rec_score:.1f}")

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
            recent_enr = int(em.sort_values("month")["enrollments"].iloc[-1]) if not em.empty else 0
            st.metric("Total Enrollments", f"{total_enr:,}")
            st.metric("Avg Enrollments/Month", "—" if np.isnan(avg_enr) else f"{avg_enr:.1f}")
            st.metric("Most Recent Month", f"{recent_enr:,}")
        else:
            st.metric("Total Enrollments", "—")
            st.metric("Avg Enrollments/Month", "—")
            st.metric("Most Recent Month", "—")
else:
    # Only show enrollments KPIs if available
    col = st.columns([1])[0]
    with col:
        if not enrollments_monthly.empty:
            em = enrollments_monthly.copy()
            total_enr = int(em["enrollments"].sum()) if not em.empty else 0
            avg_enr = float(em["enrollments"].mean()) if not em.empty else np.nan
            recent_enr = int(em.sort_values("month")["enrollments"].iloc[-1]) if not em.empty else 0
            st.metric("Total Enrollments", f"{total_enr:,}")
            st.metric("Avg Enrollments/Month", "—" if np.isnan(avg_enr) else f"{avg_enr:.1f}")
            st.metric("Most Recent Month", f"{recent_enr:,}")

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
        )
        vol_fig.update_layout(barmode="stack", legend_title_text="Source", xaxis_title=None)
        st.plotly_chart(vol_fig, use_container_width=True)
    else:
        st.info("No rows for the selected filters to plot monthly volume.")

    # 2) On-time rate by month
    sla = (
        f.assign(month=f["_trend_month"])\
         .groupby("month", dropna=False)["On Time (<= Deadline)"].mean().reset_index(name="on_time_rate")
    )
    if not sla.empty:
        sla_fig = px.line(sla, x="month", y="on_time_rate", markers=True,
                          title="On-Time Rate by Month",
                          labels={"on_time_rate": "On-Time Rate"})
        sla_fig.update_yaxes(tickformat=",.0%", range=[0,1])
        st.plotly_chart(sla_fig, use_container_width=True)

    # 3) Avg days to submit by month
    speed = (
        f.assign(month=f["_trend_month"])\
         .groupby("month", dropna=False)["Days to Submit"].mean().reset_index(name="avg_days_to_submit")
    )
    if not speed.empty and speed["avg_days_to_submit"].notna().any():
        sp_fig = px.line(speed, x="month", y="avg_days_to_submit", markers=True,
                         title="Average Days to Submit by Month",
                         labels={"avg_days_to_submit": "Days"})
        st.plotly_chart(sp_fig, use_container_width=True)

    # 4) Status breakdown by team
    if "Team Name" in f.columns and "Case Status" in f.columns and not f.empty:
        by_team = f.groupby(["Team Name", "Case Status"], dropna=False).size().reset_index(name="count")
        team_fig = px.bar(by_team, x="Team Name", y="count", color="Case Status",
                          title="Case Status by Team", barmode="stack")
        team_fig.update_layout(xaxis_title="Team", yaxis_title="Cases")
        st.plotly_chart(team_fig, use_container_width=True)

    # 5) Recording quality distribution
    if "Recording Score" in f.columns and f["Recording Score"].notna().any():
        rec_fig = px.histogram(f, x="Recording Score", nbins=20, title="Recording Score Distribution")
        st.plotly_chart(rec_fig, use_container_width=True)

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
        r_fig.add_bar(x=ratio_df["month"], y=ratio_df["complaints"], name="Complaints")
        r_fig.add_scatter(x=ratio_df["month"], y=ratio_df["enrollments"], name="Enrollments", mode="lines+markers", yaxis="y2")
        r_fig.update_layout(
            title="Complaints vs Enrollments (Dual Axis)",
            yaxis=dict(title="Complaints"),
            yaxis2=dict(title="Enrollments", overlaying="y", side="right"),
            xaxis_title=None,
            legend_title_text="Series",
        )
        st.plotly_chart(r_fig, use_container_width=True)

        rr_fig = px.line(ratio_df, x="month", y="complaints_per_1000_enrollments", markers=True,
                         title="Complaints per 1,000 Enrollments")
        st.plotly_chart(rr_fig, use_container_width=True)
    else:
        if enrollments_monthly.empty:
            st.info("Upload your enrollments file to compute ratios.")
        else:
            st.info("Not enough complaints data after filters to compute monthly totals.")
else:
    if not enrollments_monthly.empty:
        st.info("Upload your complaints file to compute ratios.")

# -------------------------
# Agent Leaderboard
# -------------------------
st.subheader("Agent Leaderboard (Complaints)")
if has_complaints_data and "Agent Name" in f.columns and not f.empty:
    grp = f.groupby("Agent Name", dropna=False)
    leaderboard = pd.DataFrame({
        "Cases": grp.size(),
        "On-Time Rate": grp["On Time (<= Deadline)"].mean(),
        "Avg Days to Submit": grp["Days to Submit"].mean(),
        "Avg Days Late": grp["Days Late"].mean(),
        "CTM Share": grp["Complaint Source"].apply(lambda s: (s == "CMS (CTM)").mean()),
        "Remedial Signed Rate": grp.apply(lambda g: to_bool(g.get("Remedial Signed", pd.Series())).mean() if "Remedial Signed" in g.columns else np.nan),
        "Training Completion Rate": grp.apply(lambda g: to_bool(g.get("Course Completed", pd.Series())).mean() if "Course Completed" in g.columns else np.nan),
    }).reset_index().sort_values(["Cases", "On-Time Rate"], ascending=[False, False])

    st.dataframe(leaderboard, use_container_width=True)

    csv_bytes = leaderboard.to_csv(index=False).encode("utf-8")
    st.download_button("Download Leaderboard (CSV)", data=csv_bytes, file_name="leaderboard.csv", mime="text/csv")
else:
    st.info("Upload your complaints file to see the leaderboard.")

# -------------------------
# Data Quality Checks
# -------------------------
st.subheader("Data Quality Checks (Complaints)")
if has_complaints_data and not f.empty:
    issues = []
    if f["Deadline Date"].isna().any():
        issues.append(f"Missing Deadline Date: {int(f['Deadline Date'].isna().sum())} rows")
    if safe_col(f, "Response Submited") and f["Response Submited"].isna().any():
        issues.append(f"Missing Response Submited: {int(f['Response Submited'].isna().sum())} rows")
    if f["Date of Occurence"].isna().any():
        issues.append(f"Missing Date of Occurence: {int(f['Date of Occurence'].isna().sum())} rows")
    # New: check received date coverage (any known variant present)
    for rc_col in ["Date Complaint Received", "Complaint Received Date", "Date Received"]:
        if safe_col(f, rc_col) and f[rc_col].isna().any():
            issues.append(f"Missing {rc_col}: {int(f[rc_col].isna().sum())} rows")

    if issues:
        st.warning("\n".join(["• " + x for x in issues]))
    else:
        st.success("No major data completeness issues detected for key date fields.")
else:
    st.info("Upload your complaints file to run data quality checks.")

# -------------------------
# Raw/Filtered Data + Export
# -------------------------
st.subheader("Filtered Complaint Rows")
if has_complaints_data:
    st.dataframe(f, use_container_width=True, height=420)

    raw_csv = f.to_csv(index=False).encode("utf-8")
    st.download_button("Download Filtered Data (CSV)", data=raw_csv, file_name="filtered_complaints_rows.csv", mime="text/csv")
