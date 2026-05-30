from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


BASE = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE / "Data" / "output"

FINAL_CENTRAL_FILE = OUTPUT_DIR / "E_CENTRALE_ENG_RSQ.csv"
V2_CENTRAL_FILE = OUTPUT_DIR / "E_CENTRALE_ENG_RSQ_v2.csv"


st.set_page_config(
    page_title="Credit Risk Dashboard",
    page_icon="📊",
    layout="wide",
)


def load_central_table() -> pd.DataFrame:
    if FINAL_CENTRAL_FILE.exists():
        path = FINAL_CENTRAL_FILE
    elif V2_CENTRAL_FILE.exists():
        path = V2_CENTRAL_FILE
    else:
        raise FileNotFoundError(
            "No central table found. Expected E_CENTRALE_ENG_RSQ.csv or E_CENTRALE_ENG_RSQ_v2.csv."
        )

    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")

    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.upper()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )

    df["SOURCE_FILE"] = path.name
    return df


def to_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype("string")
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False),
        errors="coerce",
    ).fillna(0)


def clean_text(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .replace({"": pd.NA, "<NA>": pd.NA, "nan": pd.NA, "None": pd.NA})
    )


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    numeric_cols = [
        "MONT_DEBL",
        "ENCOURS",
        "IMPAYE_PRINCIPAL",
        "IMPAYE_INTERET",
        "MONT_ENCR_CRDT",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = to_number(df[col])
        else:
            df[col] = 0

    text_cols = [
        "REF_CRDT",
        "CLIENT_ID",
        "SEGMENT_CREDIT",
        "CODE_ENG_BCT",
        "ID_ACTIVITE",
        "SECTOR",
        "CATG",
        "TYPE_ENG_BIAT",
        "DAT_SIT",
    ]

    for col in text_cols:
        if col in df.columns:
            df[col] = clean_text(df[col])
        else:
            df[col] = pd.NA

    df["TOTAL_UNPAID"] = df["IMPAYE_PRINCIPAL"] + df["IMPAYE_INTERET"]

    return df


def format_amount(value: float) -> str:
    return f"{value:,.3f}".replace(",", " ")


def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")

    filtered = df.copy()

    if "DAT_SIT" in filtered.columns:
        dates = sorted(filtered["DAT_SIT"].dropna().unique().tolist())
        selected_dates = st.sidebar.multiselect(
            "Situation date",
            options=dates,
            default=dates,
        )
        if selected_dates:
            filtered = filtered[filtered["DAT_SIT"].isin(selected_dates)]

    if "SEGMENT_CREDIT" in filtered.columns:
        segments = sorted(filtered["SEGMENT_CREDIT"].dropna().unique().tolist())
        selected_segments = st.sidebar.multiselect(
            "Credit segment",
            options=segments,
            default=segments,
        )
        if selected_segments:
            filtered = filtered[filtered["SEGMENT_CREDIT"].isin(selected_segments)]

    if "CODE_ENG_BCT" in filtered.columns:
        codes = sorted(filtered["CODE_ENG_BCT"].dropna().unique().tolist())
        selected_codes = st.sidebar.multiselect(
            "Credit-form code",
            options=codes,
            default=codes,
        )
        if selected_codes:
            filtered = filtered[filtered["CODE_ENG_BCT"].isin(selected_codes)]

    if "ID_ACTIVITE" in filtered.columns:
        activities = sorted(filtered["ID_ACTIVITE"].dropna().unique().tolist())
        selected_activities = st.sidebar.multiselect(
            "Activity code",
            options=activities,
            default=activities,
        )
        if selected_activities:
            filtered = filtered[filtered["ID_ACTIVITE"].isin(selected_activities)]

    return filtered


def kpi_cards(df: pd.DataFrame) -> None:
    total_exposure = df["MONT_ENCR_CRDT"].sum()
    total_encours = df["ENCOURS"].sum()
    total_unpaid = df["TOTAL_UNPAID"].sum()
    unpaid_ratio = (total_unpaid / total_exposure * 100) if total_exposure > 0 else 0

    credits_with_unpaid = df.loc[df["TOTAL_UNPAID"] > 0, "REF_CRDT"].nunique()
    customers_with_unpaid = df.loc[df["TOTAL_UNPAID"] > 0, "CLIENT_ID"].nunique()

    c1, c2, c3 = st.columns(3)
    c4, c5, c6 = st.columns(3)

    c1.metric("Total Credit Exposure", format_amount(total_exposure))
    c2.metric("Total Outstanding Amount", format_amount(total_encours))
    c3.metric("Total Unpaid Amount", format_amount(total_unpaid))

    c4.metric("Unpaid Exposure Ratio", f"{unpaid_ratio:.2f}%")
    c5.metric("Credits with Unpaid Amounts", f"{credits_with_unpaid:,}")
    c6.metric("Customers with Unpaid Amounts", f"{customers_with_unpaid:,}")


def exposure_composition_chart(df: pd.DataFrame) -> None:
    values = pd.DataFrame(
        {
            "Component": ["Outstanding Exposure", "Unpaid Principal", "Unpaid Interest"],
            "Amount": [
                df["ENCOURS"].sum(),
                df["IMPAYE_PRINCIPAL"].sum(),
                df["IMPAYE_INTERET"].sum(),
            ],
        }
    )

    fig = px.bar(
        values,
        x="Component",
        y="Amount",
        text_auto=".3s",
        title="Exposure Composition",
    )

    fig.update_layout(yaxis_title="Amount", xaxis_title="")
    st.plotly_chart(fig, use_container_width=True)


def exposure_by_segment_chart(df: pd.DataFrame) -> None:
    grouped = (
        df.groupby("SEGMENT_CREDIT", dropna=False, as_index=False)["MONT_ENCR_CRDT"]
        .sum()
        .sort_values("MONT_ENCR_CRDT", ascending=False)
    )

    fig = px.pie(
        grouped,
        names="SEGMENT_CREDIT",
        values="MONT_ENCR_CRDT",
        title="Exposure by Segment",
        hole=0.4,
    )

    st.plotly_chart(fig, use_container_width=True)


def exposure_by_code_chart(df: pd.DataFrame) -> None:
    grouped = (
        df.groupby("CODE_ENG_BCT", dropna=False, as_index=False)["MONT_ENCR_CRDT"]
        .sum()
        .sort_values("MONT_ENCR_CRDT", ascending=False)
    )

    fig = px.bar(
        grouped,
        x="CODE_ENG_BCT",
        y="MONT_ENCR_CRDT",
        text_auto=".3s",
        title="Exposure by Credit-Form Code",
    )

    fig.update_layout(xaxis_title="Credit-form code", yaxis_title="Exposure")
    st.plotly_chart(fig, use_container_width=True)


def top_customers_tables(df: pd.DataFrame) -> None:
    customer_exposure = (
        df.groupby("CLIENT_ID", dropna=False, as_index=False)
        .agg(
            TOTAL_EXPOSURE=("MONT_ENCR_CRDT", "sum"),
            TOTAL_UNPAID=("TOTAL_UNPAID", "sum"),
            NUMBER_OF_CREDITS=("REF_CRDT", "nunique"),
        )
        .sort_values("TOTAL_EXPOSURE", ascending=False)
    )

    left, right = st.columns(2)

    with left:
        st.subheader("Top 10 Customers by Exposure")
        st.dataframe(
            customer_exposure.head(10),
            use_container_width=True,
            hide_index=True,
        )

    with right:
        st.subheader("Top 10 Customers by Unpaid Amount")
        st.dataframe(
            customer_exposure.sort_values("TOTAL_UNPAID", ascending=False).head(10),
            use_container_width=True,
            hide_index=True,
        )


def concentration_kpi(df: pd.DataFrame) -> None:
    total_exposure = df["MONT_ENCR_CRDT"].sum()

    customer_exposure = (
        df.groupby("CLIENT_ID", dropna=False)["MONT_ENCR_CRDT"]
        .sum()
        .sort_values(ascending=False)
    )

    top_10_exposure = customer_exposure.head(10).sum()
    concentration_ratio = (
        top_10_exposure / total_exposure * 100 if total_exposure > 0 else 0
    )

    st.metric(
        "Top 10 Customer Exposure Concentration",
        f"{concentration_ratio:.2f}%",
        help="Share of total retained exposure held by the ten largest customers in the filtered perimeter.",
    )


def activity_breakdown(df: pd.DataFrame) -> None:
    activity = (
        df.groupby("ID_ACTIVITE", dropna=False, as_index=False)["MONT_ENCR_CRDT"]
        .sum()
        .sort_values("MONT_ENCR_CRDT", ascending=False)
        .head(15)
    )

    fig = px.bar(
        activity,
        x="ID_ACTIVITE",
        y="MONT_ENCR_CRDT",
        text_auto=".3s",
        title="Top Activities by Exposure",
    )

    fig.update_layout(xaxis_title="Activity code", yaxis_title="Exposure")
    st.plotly_chart(fig, use_container_width=True)


def data_quality_snapshot(df: pd.DataFrame) -> None:
    st.subheader("Data Quality Snapshot")

    quality = pd.DataFrame(
        [
            {"Control": "Rows in filtered data", "Value": len(df)},
            {"Control": "Distinct credit references", "Value": df["REF_CRDT"].nunique()},
            {"Control": "Distinct customers", "Value": df["CLIENT_ID"].nunique()},
            {"Control": "Rows with missing CODE_ENG_BCT", "Value": int(df["CODE_ENG_BCT"].isna().sum())},
            {"Control": "Rows with zero total exposure", "Value": int((df["MONT_ENCR_CRDT"] == 0).sum())},
            {"Control": "Rows with unpaid amounts", "Value": int((df["TOTAL_UNPAID"] > 0).sum())},
        ]
    )

    st.dataframe(quality, use_container_width=True, hide_index=True)


def main() -> None:
    st.title("Credit Risk Data Centralization Dashboard")
    st.caption(
        "Internal analytical view based on E_CENTRALE_ENG_RSQ. "
        "The dashboard describes the processed credit-risk perimeter and does not represent a full prudential risk model."
    )

    raw = load_central_table()
    df = prepare_data(raw)
    st.info(f"Loaded source file: {df['SOURCE_FILE'].iloc[0] if not df.empty else 'No file loaded'}")
    st.write("Rows loaded:", len(df))
    st.write("Columns:", list(df.columns))

with st.expander("Raw data preview"):
    st.dataframe(df.head(20), use_container_width=True)
    st.sidebar.caption(f"Source file: {df['SOURCE_FILE'].iloc[0]}")

    filtered = filter_dataframe(df)

    if filtered.empty:
    st.warning("No data available for the selected filters.")
    return

    st.header("Risk Overview")
    kpi_cards(filtered)

    st.divider()

    st.header("Exposure Analysis")
    concentration_kpi(filtered)

    col1, col2 = st.columns(2)

    with col1:
        exposure_composition_chart(filtered)

    with col2:
        exposure_by_segment_chart(filtered)

    exposure_by_code_chart(filtered)

    st.divider()

    st.header("Concentration and Activity Analysis")
    top_customers_tables(filtered)
    activity_breakdown(filtered)

    st.divider()

    st.header("Validation and Data Preview")
    data_quality_snapshot(filtered)

    with st.expander("Preview central table"):
        st.dataframe(filtered.head(200), use_container_width=True, hide_index=True)

    csv = filtered.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        label="Download filtered data",
        data=csv,
        file_name="filtered_credit_risk_data.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()