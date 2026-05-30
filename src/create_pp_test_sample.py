from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[1]

CENTRAL_FILE = BASE / "Data" / "output" / "E_CENTRALE_ENG_RSQ_v2.csv"
SEGMENT_FORCE_FILE = BASE / "Data" / "parameters" / "P_SEGMENT_FORCE.csv"

N_PP_SAMPLE = 100


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=None, engine="python", dtype=str, encoding="utf-8-sig")


def clean_headers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.upper()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )
    return df


central = clean_headers(read_csv(CENTRAL_FILE))
segment_force = clean_headers(read_csv(SEGMENT_FORCE_FILE))

central["MONT_ENCR_CRDT_NUM"] = pd.to_numeric(
    central["MONT_ENCR_CRDT"].astype(str).str.replace(",", "."),
    errors="coerce"
).fillna(0)

eligible = central[
    (central["MONT_ENCR_CRDT_NUM"] > 0)
    & (central["CODE_ENG_BCT"].notna())
    & (central["CODE_ENG_BCT"].astype(str).str.strip() != "")
].copy()

sample_refs = eligible["REF_CRDT"].drop_duplicates().head(N_PP_SAMPLE)

pp_sample = pd.DataFrame({
    "REF_CRDT": sample_refs,
    "SEGMENT_RSQ_FINAL": "PP",
    "STATUT_DECL": "1",
})

# Remove existing rows for these REF_CRDT to avoid duplicates/conflicts.
segment_force = segment_force[
    ~segment_force["REF_CRDT"].isin(pp_sample["REF_CRDT"])
].copy()

updated_segment_force = pd.concat(
    [segment_force, pp_sample],
    ignore_index=True
)

updated_segment_force.to_csv(
    SEGMENT_FORCE_FILE,
    index=False,
    encoding="utf-8-sig"
)

print(f"Added/updated {len(pp_sample)} REF_CRDT as PP in P_SEGMENT_FORCE.")
print(f"Updated file: {SEGMENT_FORCE_FILE}")