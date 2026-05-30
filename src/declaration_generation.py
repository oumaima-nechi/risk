from pathlib import Path
import pandas as pd


BASE = Path(__file__).resolve().parents[1]

OUTPUT_DIR = BASE / "Data" / "output"
QUALITY_DIR = BASE / "Data" / "quality"

CENTRAL_FILE = OUTPUT_DIR / "E_CENTRALE_ENG_RSQ_v2.csv"

INDIV_FILE = OUTPUT_DIR / "E_INDIV_CONTINU.csv"
GLB_FILE = OUTPUT_DIR / "E_GLB_CONTINU.csv"
QUALITY_FILE = QUALITY_DIR / "declaration_generation_quality.csv"


def read_central_table() -> pd.DataFrame:
    if not CENTRAL_FILE.exists():
        raise FileNotFoundError(f"Central table not found: {CENTRAL_FILE}")

    df = pd.read_csv(CENTRAL_FILE, dtype=str, encoding="utf-8-sig")

    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.upper()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )

    return df


def clean_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def clean_upper(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.upper()


def clean_flag(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .str.replace(".0", "", regex=False)
    )


def to_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype("string")
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False),
        errors="coerce",
    ).fillna(0)


def is_not_empty(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.strip()
    return (
        cleaned.notna()
        & ~cleaned.isin(["", "<NA>", "nan", "NaN", "None", "NONE"])
    )


def is_empty(series: pd.Series) -> pd.Series:
    return ~is_not_empty(series)


def prepare_central(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["SEGMENT_CREDIT"] = clean_upper(df["SEGMENT_CREDIT"])
    df["STATUT_DECLR"] = clean_flag(df["STATUT_DECLR"])
    df["ID_BCT"] = clean_text(df["ID_BCT"])
    df["TYPE_ID_BCT"] = clean_text(df["TYPE_ID_BCT"])
    df["CODE_ENG_BCT"] = clean_text(df["CODE_ENG_BCT"])
    df["ID_ACTIVITE"] = clean_text(df["ID_ACTIVITE"])
    df["DAT_SIT"] = clean_text(df["DAT_SIT"])
    df["MONT_ENCR_CRDT"] = to_number(df["MONT_ENCR_CRDT"])

    return df


def build_indiv_table(central: pd.DataFrame) -> pd.DataFrame:
    indiv_source = central[
        (central["SEGMENT_CREDIT"] == "PM")
        & is_not_empty(central["ID_BCT"])
        & (central["STATUT_DECLR"] == "1")
        & is_not_empty(central["CODE_ENG_BCT"])
        & (central["MONT_ENCR_CRDT"] > 0)
    ].copy()

    indiv_source["CODE_BCT"] = indiv_source["CODE_ENG_BCT"]
    indiv_source["CODE_OPERATION"] = "N"
    indiv_source["DATE_MODIFICATION"] = pd.NA

    indiv = indiv_source[
        [
            "ID_BCT",
            "TYPE_ID_BCT",
            "CODE_BCT",
            "MONT_ENCR_CRDT",
            "CODE_OPERATION",
            "STATUT_DECLR",
            "DATE_MODIFICATION",
            "DAT_SIT",
        ]
    ].copy()

    return indiv


def build_glb_table(central: pd.DataFrame) -> pd.DataFrame:
    glb_source = central[
        (central["SEGMENT_CREDIT"] == "PP")
        & is_empty(central["ID_BCT"])
        & (central["STATUT_DECLR"] == "1")
        & is_not_empty(central["CODE_ENG_BCT"])
        & is_not_empty(central["ID_ACTIVITE"])
        & (central["MONT_ENCR_CRDT"] > 0)
    ].copy()

    glb = (
        glb_source
        .groupby(["ID_ACTIVITE", "CODE_ENG_BCT", "DAT_SIT"], as_index=False)["MONT_ENCR_CRDT"]
        .sum()
    )

    glb["CODE_BCT"] = glb["CODE_ENG_BCT"]
    glb["CODE_OPERATION"] = "N"
    glb["DATE_MODIFICATION"] = pd.NA
    glb["STATUT_DECLR"] = "1"

    glb = glb[
        [
            "ID_ACTIVITE",
            "CODE_BCT",
            "MONT_ENCR_CRDT",
            "CODE_OPERATION",
            "DATE_MODIFICATION",
            "STATUT_DECLR",
            "DAT_SIT",
        ]
    ].copy()

    return glb


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    QUALITY_DIR.mkdir(parents=True, exist_ok=True)

    central = prepare_central(read_central_table())

    indiv = build_indiv_table(central)
    glb = build_glb_table(central)

    indiv.to_csv(INDIV_FILE, index=False, encoding="utf-8-sig")
    glb.to_csv(GLB_FILE, index=False, encoding="utf-8-sig")

    quality = pd.DataFrame(
        [
            {"control": "Central table rows", "value": len(central)},
            {"control": "PM central rows", "value": int((central["SEGMENT_CREDIT"] == "PM").sum())},
            {"control": "PP central rows", "value": int((central["SEGMENT_CREDIT"] == "PP").sum())},
            {"control": "Rows with ID_BCT not empty", "value": int(is_not_empty(central["ID_BCT"]).sum())},
            {"control": "Rows with ID_BCT empty", "value": int(is_empty(central["ID_BCT"]).sum())},
            {"control": "Rows with CODE_ENG_BCT not empty", "value": int(is_not_empty(central["CODE_ENG_BCT"]).sum())},
            {"control": "Rows with ID_ACTIVITE not empty", "value": int(is_not_empty(central["ID_ACTIVITE"]).sum())},
            {"control": "Rows with MONT_ENCR_CRDT > 0", "value": int((central["MONT_ENCR_CRDT"] > 0).sum())},
            {"control": "Individual preparation rows", "value": len(indiv)},
            {"control": "Global preparation rows", "value": len(glb)},
            {"control": "Individual total amount", "value": float(indiv["MONT_ENCR_CRDT"].sum()) if not indiv.empty else 0},
            {"control": "Global total amount", "value": float(glb["MONT_ENCR_CRDT"].sum()) if not glb.empty else 0},
            {"control": "Central total amount", "value": float(central["MONT_ENCR_CRDT"].sum())},
        ]
    )

    quality.to_csv(QUALITY_FILE, index=False, encoding="utf-8-sig")

    print("Declaration preparation complete.")
    print(f"Individual table rows: {len(indiv)}")
    print(f"Global table rows: {len(glb)}")
    print(f"Individual table saved to: {INDIV_FILE}")
    print(f"Global table saved to: {GLB_FILE}")
    print(f"Quality report saved to: {QUALITY_FILE}")


if __name__ == "__main__":
    main()