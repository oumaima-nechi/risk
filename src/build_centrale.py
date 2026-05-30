from pathlib import Path
import hashlib
import pandas as pd


BASE = Path(__file__).resolve().parents[1]

RAW_DIR = BASE / "Data" / "Raw"
PARAM_DIR = BASE / "Data" / "parameters"
OUTPUT_DIR = BASE / "Data" / "output"
QUALITY_DIR = BASE / "Data" / "quality"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
QUALITY_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "customer": RAW_DIR / "E_CUSTOMER.csv",
    "contrat": RAW_DIR / "E_CONTRAT_CREDIT.csv",
    "encours": RAW_DIR / "E_ENCOURS.csv",
    "p_code_doc": PARAM_DIR / "P_CODE_DOC_BCT.csv",
    "p_segment_force": PARAM_DIR / "P_SEGMENT_FORCE.csv",
    "p_forme_credit": PARAM_DIR / "P_FORME_CREDIT_BCT.csv",
}


def read_csv_flexible(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    for encoding in ["utf-8-sig", "utf-8", "latin-1"]:
        try:
            return pd.read_csv(path, sep=None, engine="python", dtype=str, encoding=encoding)
        except Exception:
            continue

    raise RuntimeError(f"Could not read file: {path}")


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


def clean_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        df[col] = df[col].astype("string").str.strip()
        df[col] = df[col].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    return df


def load_table(path: Path) -> pd.DataFrame:
    return clean_string_columns(clean_headers(read_csv_flexible(path)))


def to_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype("string")
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False),
        errors="coerce",
    )


def generate_synthetic_bct_id(client_id: str) -> str:
    if pd.isna(client_id) or str(client_id).strip() == "":
        client_id = "UNKNOWN"

    digest = hashlib.md5(str(client_id).encode("utf-8")).hexdigest()
    numeric = int(digest, 16) % 10**12
    return "BCT" + str(numeric).zfill(12)


def select_id_doc(row: pd.Series) -> pd.Series:
    priority = [
        ("CIN", "CIN"),
        ("PASS", "PASS"),
        ("CART_SEJR", "CART_SEJR"),
        ("VISA", "VISA"),
        ("IDNT_UNQ", "IDNT_UNQ"),
        ("IDNT_FISC", "IDNT_FISC"),
        ("MATR_FISC", "MATR_FISC"),
    ]

    for field, doc_type in priority:
        value = row.get(field)
        if pd.notna(value) and str(value).strip() != "":
            return pd.Series({"ID_DOC": value, "DOC_TYPE": doc_type})

    return pd.Series({"ID_DOC": pd.NA, "DOC_TYPE": pd.NA})


def split_values(value: str) -> list[str]:
    if pd.isna(value):
        return []
    return [
        item.strip()
        for item in str(value).replace(";", "|").split("|")
        if item.strip() != ""
    ]


def normalize_component(value: str) -> str | None:
    if pd.isna(value):
        return None

    value = str(value).strip().upper()

    mapping = {
        "ENCOURS": "ENCOURS",
        "ENCOURS_PRIN": "ENCOURS",
        "ENCOURS_IMP_PRIN": "IMPAYE_PRINCIPAL",
        "ENCOURS_IMP_INT": "IMPAYE_INTERET",
    }

    return mapping.get(value)


def format_code_bct(value: str):
    if pd.isna(value):
        return pd.NA

    value = str(value).strip()

    if value.endswith(".0"):
        value = value[:-2]

    if value.isdigit():
        return value.zfill(4)

    return value


def explode_p_forme_credit(p_forme_credit: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, row in p_forme_credit.iterrows():
        code_bct = format_code_bct(row.get("CODE_BCT"))
        segment = str(row.get("SEGMENT_CRDT")).strip().upper()

        type_eng_values = split_values(row.get("TYPE_ENG_BIAT"))

        type_encours_groups = [
            split_values(group)
            for group in str(row.get("TYPE_ENCOURS")).split("|")
        ]

        declaration_groups = [
            item.strip()
            for item in str(row.get("MONTANT_DECLARATION")).split("|")
            if item.strip() != ""
        ]

        if len(declaration_groups) == 1 and len(type_encours_groups) > 1:
            declaration_groups = declaration_groups * len(type_encours_groups)

        for index, type_encours_values in enumerate(type_encours_groups):
            if index >= len(declaration_groups):
                continue

            target_component = normalize_component(declaration_groups[index])

            if target_component is None:
                continue

            for type_eng in type_eng_values:
                for type_encours in type_encours_values:
                    rows.append(
                        {
                            "TYPE_ENG_BIAT": str(type_eng).strip(),
                            "TYPE_ENCOURS": str(type_encours).strip(),
                            "SEGMENT_CRDT": segment,
                            "CODE_ENG_BCT": code_bct,
                            "TARGET_COMPONENT": target_component,
                        }
                    )

    return pd.DataFrame(rows).drop_duplicates()


def main() -> None:
    print("Loading source files...")

    customer = load_table(FILES["customer"])
    contrat = load_table(FILES["contrat"])
    encours = load_table(FILES["encours"])
    p_code_doc = load_table(FILES["p_code_doc"])
    p_segment_force = load_table(FILES["p_segment_force"])
    p_forme_credit = load_table(FILES["p_forme_credit"])

    print(f"E_CUSTOMER: {len(customer)} rows")
    print(f"E_CONTRAT_CREDIT: {len(contrat)} rows")
    print(f"E_ENCOURS: {len(encours)} rows")
    print(f"P_CODE_DOC_BCT: {len(p_code_doc)} rows")
    print(f"P_SEGMENT_FORCE: {len(p_segment_force)} rows")
    print(f"P_FORME_CREDIT_BCT: {len(p_forme_credit)} rows")

    # 1. Active exposures
    encours["STATUS"] = encours["STATUS"].astype("string").str.upper().str.strip()
    active_encours = encours[encours["STATUS"] == "A"].copy()
    print(f"Active exposure rows: {len(active_encours)}")

    # 2. Join exposure with contracts
    encours_contract = active_encours.merge(
        contrat,
        on="REF_CONT",
        how="left",
        suffixes=("_ENCOURS", "_CONTRAT"),
    )

    unmatched_contract_rows = encours_contract["REF_CRDT"].isna().sum()
    print(f"Active exposure rows without matching contract: {unmatched_contract_rows}")

    encours_contract = encours_contract[encours_contract["REF_CRDT"].notna()].copy()

    # 3. One row per REF_CRDT
    needed_fields = [
        "REF_CRDT",
        "REF_CONT",
        "CLIENT_ID",
        "VAL_DATE",
        "MATR_DATE",
        "CATG",
        "TYPE_ENG",
        "REF_TITR",
        "MONT_DEBL",
        "DAT_SIT",
    ]

    central = (
        encours_contract[needed_fields]
        .sort_values(["REF_CRDT", "DAT_SIT"])
        .groupby("REF_CRDT", as_index=False)
        .agg(
            {
                "REF_CONT": "first",
                "CLIENT_ID": "first",
                "VAL_DATE": "first",
                "MATR_DATE": "first",
                "CATG": "first",
                "TYPE_ENG": "first",
                "REF_TITR": "first",
                "MONT_DEBL": "first",
                "DAT_SIT": "max",
            }
        )
    )

    central["REF_TITR"] = central["REF_TITR"].fillna(central["REF_CRDT"])
    print(f"Central records after grouping by REF_CRDT: {len(central)}")

    # 4. Join customer
    customer_fields = [
        "CLIENT_ID",
        "CUST_TYPE",
        "SECT_ID",
        "INDS_ID",
        "ACTV_ID",
        "NUM_BCT",
        "CIN",
        "PASS",
        "CART_SEJR",
        "VISA",
        "IDNT_UNQ",
        "IDNT_FISC",
        "MATR_FISC",
    ]

    customer_light = customer[customer_fields].drop_duplicates(subset=["CLIENT_ID"])

    central = central.merge(customer_light, on="CLIENT_ID", how="left")

    unmatched_customer_rows = central["CUST_TYPE"].isna().sum()
    print(f"Central records without matching customer information: {unmatched_customer_rows}")

    # 5. Core fields
    central["TYPE_ENG_BIAT"] = central["TYPE_ENG"]
    central["ID_ACTIVITE"] = central["ACTV_ID"]
    central["SECTOR"] = central["SECT_ID"]
    central["MONT_DEBL"] = to_number(central["MONT_DEBL"])



    # 7. Segment and declaration status
         # 7. Segment and declaration status
    p_segment_force = p_segment_force.drop_duplicates(subset=["REF_CRDT"])

    central = central.merge(
        p_segment_force[["REF_CRDT", "SEGMENT_RSQ_FINAL", "STATUT_DECL"]],
        on="REF_CRDT",
        how="left",
    )

    # Temporary fallback based on the sample P_FORME_CREDIT_BCT:
    # PP: 721, 722, 724
    # PM: 524, 526
    # Default: PM
    type_eng_segment_map = {
        "721": "PP",
        "722": "PP",
        "724": "PP",
        "524": "PM",
        "526": "PM",
    }

    central["TYPE_ENG_CLEAN"] = (
        central["TYPE_ENG"]
        .astype("string")
        .str.strip()
    )

    central["SEGMENT_CREDIT"] = central["SEGMENT_RSQ_FINAL"]

    missing_segment = (
        central["SEGMENT_CREDIT"].isna()
        | (central["SEGMENT_CREDIT"].astype("string").str.strip() == "")
    )

    central.loc[missing_segment, "SEGMENT_CREDIT"] = central.loc[
        missing_segment, "TYPE_ENG_CLEAN"
    ].map(type_eng_segment_map)

    central["SEGMENT_CREDIT"] = (
        central["SEGMENT_CREDIT"]
        .fillna("PM")
        .astype("string")
        .str.upper()
        .str.strip()
    )

    central["STATUT_DECLR"] = central["STATUT_DECL"].fillna("1")

    # Synthetic ID_BCT is generated only for PM records.
    # PP records keep ID_BCT null so they can feed the global preparation table.
    central["ID_BCT"] = pd.NA
    central["TYPE_ID_BCT"] = pd.NA

    pm_mask = central["SEGMENT_CREDIT"] == "PM"

    central.loc[pm_mask, "ID_BCT"] = central.loc[
        pm_mask, "CLIENT_ID"
    ].apply(generate_synthetic_bct_id)

    central.loc[pm_mask, "TYPE_ID_BCT"] = "01"
   

    central["SEGMENT_CREDIT"] = central["SEGMENT_RSQ_FINAL"]

    missing_segment = central["SEGMENT_CREDIT"].isna() | (
    central["SEGMENT_CREDIT"].astype("string").str.strip() == ""
)

  

    central["SEGMENT_CREDIT"] = (
    central["SEGMENT_CREDIT"]
    .astype("string")
    .str.upper()
    .str.strip()
)

    central["STATUT_DECLR"] = central["STATUT_DECL"].fillna("1")

# Synthetic ID_BCT is generated only for PM records.
# PP records keep ID_BCT null so that they can feed the global preparation table.
    central["ID_BCT"] = pd.NA
    central["TYPE_ID_BCT"] = pd.NA

    pm_mask = central["SEGMENT_CREDIT"] == "PM"

    central.loc[pm_mask, "ID_BCT"] = central.loc[
    pm_mask, "CLIENT_ID"
].apply(generate_synthetic_bct_id)

    central.loc[pm_mask, "TYPE_ID_BCT"] = "01"

    # 8. Monetary components and CODE_ENG_BCT
    forme_mapping = explode_p_forme_credit(p_forme_credit)

    print("\nExpanded P_FORME_CREDIT_BCT mapping:")
    print(forme_mapping.to_string(index=False))

    exposure_for_amounts = encours_contract.merge(
        central[["REF_CRDT", "SEGMENT_CREDIT"]],
        on="REF_CRDT",
        how="left",
    )

    exposure_for_amounts["SEGMENT_CREDIT"] = (
        exposure_for_amounts["SEGMENT_CREDIT"]
        .fillna("PM")
        .astype("string")
        .str.upper()
        .str.strip()
    )

    exposure_for_amounts["TYPE_ENG_BIAT"] = (
        exposure_for_amounts["TYPE_ENG"].astype("string").str.strip()
    )

    exposure_for_amounts["TYPE_ENCOURS"] = (
        exposure_for_amounts["CODE_TYP_ENCR"].astype("string").str.strip()
    )

    exposure_for_amounts["MONTANT_SOURCE_TND"] = to_number(
        exposure_for_amounts["MONT_ENCR_REDR_TND"]
    ).fillna(0)

    exposure_mapped = exposure_for_amounts.merge(
        forme_mapping,
        left_on=["TYPE_ENG_BIAT", "TYPE_ENCOURS", "SEGMENT_CREDIT"],
        right_on=["TYPE_ENG_BIAT", "TYPE_ENCOURS", "SEGMENT_CRDT"],
        how="left",
    )

        # Fallback mapping for prototype testing when P_FORME_CREDIT_BCT
    # does not cover all TYPE_ENG values in the available sample.
    fallback_component_map = {
        "1": "ENCOURS",
        "2": "ENCOURS",
        "3": "IMPAYE_PRINCIPAL",
        "4": "IMPAYE_INTERET",
    }

    exposure_mapped["TYPE_ENCOURS"] = (
        exposure_mapped["TYPE_ENCOURS"]
        .astype("string")
        .str.replace(".0", "", regex=False)
        .str.strip()
    )

    exposure_mapped["TARGET_COMPONENT"] = exposure_mapped["TARGET_COMPONENT"].fillna(
        exposure_mapped["TYPE_ENCOURS"].map(fallback_component_map)
    )

    # If CODE_ENG_BCT is still missing, assign a prototype code according to the component.
    fallback_code_map = {
        "ENCOURS": "0220",
        "IMPAYE_PRINCIPAL": "0230",
        "IMPAYE_INTERET": "0240",
    }

    exposure_mapped["CODE_ENG_BCT"] = exposure_mapped["CODE_ENG_BCT"].fillna(
        exposure_mapped["TARGET_COMPONENT"].map(fallback_code_map)
    )

    unmapped_amount_rows = exposure_mapped["TARGET_COMPONENT"].isna().sum()
    print(f"Exposure rows without monetary mapping after fallback: {unmapped_amount_rows}")

    mapped_amounts = exposure_mapped[
        exposure_mapped["TARGET_COMPONENT"].notna()
    ].copy()

    mapped_amounts = exposure_mapped[exposure_mapped["TARGET_COMPONENT"].notna()].copy()

    if mapped_amounts.empty:
        monetary_pivot = pd.DataFrame(
            {
                "REF_CRDT": central["REF_CRDT"],
                "ENCOURS": 0,
                "IMPAYE_PRINCIPAL": 0,
                "IMPAYE_INTERET": 0,
                "MONT_ENCR_CRDT": 0,
                "CODE_ENG_BCT": pd.NA,
            }
        )
    else:
        monetary_pivot = (
            mapped_amounts.groupby(["REF_CRDT", "TARGET_COMPONENT"], as_index=False)[
                "MONTANT_SOURCE_TND"
            ]
            .sum()
            .pivot(index="REF_CRDT", columns="TARGET_COMPONENT", values="MONTANT_SOURCE_TND")
            .reset_index()
        )

        for component in ["ENCOURS", "IMPAYE_PRINCIPAL", "IMPAYE_INTERET"]:
            if component not in monetary_pivot.columns:
                monetary_pivot[component] = 0

        monetary_pivot[["ENCOURS", "IMPAYE_PRINCIPAL", "IMPAYE_INTERET"]] = (
            monetary_pivot[["ENCOURS", "IMPAYE_PRINCIPAL", "IMPAYE_INTERET"]].fillna(0)
        )

        monetary_pivot["MONT_ENCR_CRDT"] = (
            monetary_pivot["ENCOURS"]
            + monetary_pivot["IMPAYE_PRINCIPAL"]
            + monetary_pivot["IMPAYE_INTERET"]
        )

        priority = {
            "ENCOURS": 1,
            "IMPAYE_PRINCIPAL": 2,
            "IMPAYE_INTERET": 3,
        }

        code_by_ref = mapped_amounts[["REF_CRDT", "CODE_ENG_BCT", "TARGET_COMPONENT"]].copy()
        code_by_ref["PRIORITY"] = code_by_ref["TARGET_COMPONENT"].map(priority)

        code_by_ref = (
            code_by_ref.sort_values(["REF_CRDT", "PRIORITY"])
            .drop_duplicates(subset=["REF_CRDT"])
            [["REF_CRDT", "CODE_ENG_BCT"]]
        )

        monetary_pivot = monetary_pivot.merge(code_by_ref, on="REF_CRDT", how="left")

    central = central.merge(
        monetary_pivot[
            [
                "REF_CRDT",
                "ENCOURS",
                "IMPAYE_PRINCIPAL",
                "IMPAYE_INTERET",
                "MONT_ENCR_CRDT",
                "CODE_ENG_BCT",
            ]
        ],
        on="REF_CRDT",
        how="left",
    )

    for component in ["ENCOURS", "IMPAYE_PRINCIPAL", "IMPAYE_INTERET", "MONT_ENCR_CRDT"]:
        central[component] = central[component].fillna(0)

    print(f"Rows with MONT_ENCR_CRDT > 0: {(central['MONT_ENCR_CRDT'] > 0).sum()}")

    # 9. ID_DOC and CODE_DOC
    id_doc_result = central.apply(select_id_doc, axis=1)
    central = pd.concat([central, id_doc_result], axis=1)

    p_code_doc["LIBL_DOC_BIAT"] = (
    p_code_doc["LIBL_DOC_BIAT"]
    .astype("string")
    .str.upper()
    .str.strip()
)

    p_code_doc["CODE_DOC_BCT"] = (
    p_code_doc["CODE_DOC_BCT"]
    .astype("string")
    .str.strip()
)
    p_code_doc["LIBL_DOC_BIAT"] = (
    p_code_doc["LIBL_DOC_BIAT"]
    .astype("string")
    .str.upper()
    .str.strip()
)

    p_code_doc["CODE_DOC_BCT"] = (
    p_code_doc["CODE_DOC_BCT"]
    .astype("string")
    .str.strip()
)

    doc_mapper = dict(
    zip(
        p_code_doc["LIBL_DOC_BIAT"],
        p_code_doc["CODE_DOC_BCT"],
    )
)

# Safety fallback based on the sample parameterization table
    doc_mapper.update({
    "VISA": "2",
    "CIN": "3",
    "PASS": "4",
})

    central["CODE_DOC"] = (
    central["DOC_TYPE"]
    .astype("string")
    .str.upper()
    .str.strip()
    .map(doc_mapper)
)
   
        # -------------------------------------------------
    # Final safety correction for SEGMENT_CREDIT and ID_BCT
    # -------------------------------------------------

    # Clean TYPE_ENG for fallback segmentation
    central["TYPE_ENG_CLEAN"] = (
        central["TYPE_ENG_BIAT"]
        .astype("string")
        .str.strip()
    )

    # If SEGMENT_CREDIT was not populated by P_SEGMENT_FORCE,
    # infer it from the temporary TYPE_ENG mapping.
    type_eng_segment_map = {
        "721": "PP",
        "722": "PP",
        "724": "PP",
        "524": "PM",
        "526": "PM",
    }

    central["SEGMENT_CREDIT"] = (
        central["SEGMENT_CREDIT"]
        .astype("string")
        .str.upper()
        .str.strip()
    )

    missing_segment = (
        central["SEGMENT_CREDIT"].isna()
        | (central["SEGMENT_CREDIT"] == "")
        | (central["SEGMENT_CREDIT"] == "<NA>")
        | (central["SEGMENT_CREDIT"] == "NAN")
    )

    central.loc[missing_segment, "SEGMENT_CREDIT"] = central.loc[
        missing_segment, "TYPE_ENG_CLEAN"
    ].map(type_eng_segment_map)

    # Final fallback: records still not classified are treated as PM
    # for prototype testing.
    central["SEGMENT_CREDIT"] = (
        central["SEGMENT_CREDIT"]
        .fillna("PM")
        .replace({"<NA>": "PM", "NAN": "PM", "": "PM"})
        .astype("string")
        .str.upper()
        .str.strip()
    )

    # Rebuild ID_BCT according to the declaration filters:
    # PM records receive synthetic ID_BCT, PP records keep ID_BCT null.
    central["ID_BCT"] = pd.NA
    central["TYPE_ID_BCT"] = pd.NA

    pm_mask = central["SEGMENT_CREDIT"] == "PM"

    central.loc[pm_mask, "ID_BCT"] = central.loc[
        pm_mask, "CLIENT_ID"
    ].apply(generate_synthetic_bct_id)

    central.loc[pm_mask, "TYPE_ID_BCT"] = "01"

    print("\nFinal segment distribution:")
    print(central["SEGMENT_CREDIT"].value_counts(dropna=False))
    print("ID_BCT not null:", central["ID_BCT"].notna().sum())
    print("ID_BCT null:", central["ID_BCT"].isna().sum())
    # 10. Output
    output_columns = [
        "REF_CRDT",
        "REF_TITR",
        "CLIENT_ID",
        "ID_BCT",
        "TYPE_ID_BCT",
        "VAL_DATE",
        "ID_ACTIVITE",
        "MATR_DATE",
        "SECTOR",
        "CATG",
        "TYPE_ENG_BIAT",
        "SEGMENT_CREDIT",
        "STATUT_DECLR",
        "CODE_ENG_BCT",
        "MONT_DEBL",
        "ENCOURS",
        "IMPAYE_PRINCIPAL",
        "IMPAYE_INTERET",
        "MONT_ENCR_CRDT",
        "ID_DOC",
        "DOC_TYPE",
        "CODE_DOC",
        "DAT_SIT",
    ]

    central_output = central[output_columns].copy()

    output_path = OUTPUT_DIR / "E_CENTRALE_ENG_RSQ_v2.csv"
    central_output.to_csv(output_path, index=False, encoding="utf-8-sig")

    quality = pd.DataFrame(
        [
            {"control": "E_CUSTOMER rows", "value": len(customer)},
            {"control": "E_CONTRAT_CREDIT rows", "value": len(contrat)},
            {"control": "E_ENCOURS rows", "value": len(encours)},
            {"control": "Active exposure rows", "value": len(active_encours)},
            {"control": "Rows after exposure-contract join", "value": len(encours_contract)},
            {"control": "Unmatched exposure-contract rows", "value": int(unmatched_contract_rows)},
            {"control": "Distinct REF_CRDT in central table", "value": len(central_output)},
            {"control": "Central rows without matching customer info", "value": int(unmatched_customer_rows)},
            {"control": "Rows with synthetic ID_BCT", "value": len(central_output)},
            {"control": "Rows with missing ID_DOC", "value": int(central_output["ID_DOC"].isna().sum())},
            {"control": "Rows with missing CODE_DOC", "value": int(central_output["CODE_DOC"].isna().sum())},
            {"control": "Rows with default PM segment", "value": int((central_output["SEGMENT_CREDIT"] == "PM").sum())},
            {"control": "Rows with STATUT_DECLR = 1", "value": int((central_output["STATUT_DECLR"].astype(str) == "1").sum())},
            {"control": "Exposure rows without monetary mapping", "value": int(unmapped_amount_rows)},
            {"control": "Rows with MONT_ENCR_CRDT > 0", "value": int((central_output["MONT_ENCR_CRDT"] > 0).sum())},
            {"control": "Total ENCOURS", "value": float(central_output["ENCOURS"].sum())},
            {"control": "Total IMPAYE_PRINCIPAL", "value": float(central_output["IMPAYE_PRINCIPAL"].sum())},
            {"control": "Total IMPAYE_INTERET", "value": float(central_output["IMPAYE_INTERET"].sum())},
            {"control": "Total MONT_ENCR_CRDT", "value": float(central_output["MONT_ENCR_CRDT"].sum())},
            {"control": "Rows with missing CODE_ENG_BCT", "value": int(central_output["CODE_ENG_BCT"].isna().sum())},
        ]
    )

    quality_path = QUALITY_DIR / "build_centrale_v2_quality.csv"
    quality.to_csv(quality_path, index=False, encoding="utf-8-sig")

    print("\nPrototype assumptions applied:")
    print("- ID_BCT was generated synthetically because BIAT has not yet alimented it.")
    print("- TYPE_ID_BCT was initialized to '01'.")
    print("- SEGMENT_CREDIT uses P_SEGMENT_FORCE when available; otherwise defaults to PM.")
    print("- STATUT_DECLR uses P_SEGMENT_FORCE when available; otherwise defaults to 1.")
    print("- Monetary components are mapped through P_FORME_CREDIT_BCT.")

    print("\nBuild complete.")
    print(f"Central table saved to: {output_path}")
    print(f"Quality report saved to: {quality_path}")


if __name__ == "__main__":
    main()