print("PROFILING SCRIPT STARTED")
from pathlib import Path
import pandas as pd

print("PROFILING SCRIPT STARTED")

BASE = Path(__file__).resolve().parents[1]

RAW_FILES = {
    "E_CUSTOMER": BASE / "Data" / "Raw" / "E_CUSTOMER.csv",
    "E_CONTRAT_CREDIT": BASE / "Data" / "Raw" / "E_CONTRAT_CREDIT.csv",
    "E_ENCOURS": BASE / "Data" / "Raw" / "E_ENCOURS.csv",
}

PARAM_FILES = {
    "P_CODE_DOC_BCT": BASE / "Data" / "parameters" / "P_CODE_DOC_BCT.csv",
    "P_FORME_CREDIT_BCT": BASE / "Data" / "parameters" / "P_FORME_CREDIT_BCT.csv",
    "P_SEGMENT_FORCE": BASE / "Data" / "parameters" / "P_SEGMENT_FORCE.csv",
    "P_SEGMENT_RISQ_CONDITION": BASE / "Data" / "parameters" / "P_SEGMENT_RISQ_CONDITION.csv",
    "P_SEGMENT_RISQ_REGLE": BASE / "Data" / "parameters" / "P_SEGMENT_RISQ_REGLE.csv",
}

QUALITY_DIR = BASE / "Data" / "quality"
QUALITY_DIR.mkdir(parents=True, exist_ok=True)


def read_csv_flexible(path: Path) -> pd.DataFrame:
    for encoding in ["utf-8-sig", "utf-8", "latin-1"]:
        try:
            return pd.read_csv(path, sep=None, engine="python", dtype=str, encoding=encoding)
        except Exception:
            pass

    raise RuntimeError(f"Could not read file: {path}")


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.upper()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )
    return df


def profile_table(name: str, path: Path) -> dict:
    print(f"Checking {name}: {path}")

    if not path.exists():
        print(f"[MISSING] {name}")
        return {
            "table_name": name,
            "status": "missing",
            "rows": 0,
            "columns": 0,
        }

    df = clean_columns(read_csv_flexible(path))

    columns_file = QUALITY_DIR / f"{name.lower()}_columns.txt"
    columns_file.write_text("\n".join(df.columns), encoding="utf-8")

    missingness = (
        df.isna()
        .sum()
        .rename("missing_values")
        .reset_index()
        .rename(columns={"index": "field"})
    )

    missingness["missing_percentage"] = (
        missingness["missing_values"] / max(len(df), 1) * 100
    ).round(2)

    missingness.to_csv(
        QUALITY_DIR / f"{name.lower()}_missingness.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(f"[LOADED] {name}: {len(df)} rows | {len(df.columns)} columns")

    return {
        "table_name": name,
        "status": "loaded",
        "rows": len(df),
        "columns": len(df.columns),
    }


def main() -> None:
    profiles = []

    print("\n--- Raw files ---")
    for name, path in RAW_FILES.items():
        profiles.append(profile_table(name, path))

    print("\n--- Parameter files ---")
    for name, path in PARAM_FILES.items():
        profiles.append(profile_table(name, path))

    summary = pd.DataFrame(profiles)
    summary_path = QUALITY_DIR / "source_profile_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\nProfiling complete.")
    print(f"Summary saved to: {summary_path}")
    print(f"Reports saved in: {QUALITY_DIR}")


if __name__ == "__main__":
    main()