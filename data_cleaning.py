"""
This script processes raw healthcare data and creates features to predict 
which members will have high healthcare costs next year.
"""

import os
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings("ignore")


# CONFIG 

TRAIN_FOLDER = r"C:\Users\Hp\Downloads\softec-26-machine-learning-competition\train\train"
TEST_FOLDER  = r"C:\Users\Hp\Downloads\softec-26-machine-learning-competition\test\test"
OUTPUT_FOLDER = r"C:\Users\Hp\Downloads\softec-26-machine-learning-competition\output"


# HELPERS 

def log(message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")


def save(df, name):
    os.makedirs(os.path.join(OUTPUT_FOLDER, "intermediates"), exist_ok=True)
    path = os.path.join(OUTPUT_FOLDER, "intermediates", f"{name}.csv")
    df.to_csv(path, index=False)
    log(f"Saved: {name}.csv  {df.shape}")


def load_data(folder, filename):
    path = os.path.join(folder, filename)
    if not os.path.exists(path):
        log(f"⚠ File not found: {filename}")
        return None
    
    df = pd.read_csv(path, low_memory=False)
    log(f"Loaded: {filename}  {df.shape}")
    return df


# Global cost caps learned from training data
cost_caps = {}


# MAIN DATA 

def process_main_data(folder, dataset_type):
    log(f"\n=== Processing main data ({dataset_type}) ===")

    df = load_data(folder, f"main_df_{dataset_type}.csv")
    if df is None:
        return None

    # Keep only year-end records
    df = df[df["MONTH"].isin([-1, 12])].copy()

    # Remove leakage
    if "NEXT_YEAR_COST" in df.columns:
        df.drop(columns="NEXT_YEAR_COST", inplace=True)
    if dataset_type == "test" and "HighCostLabel" in df.columns:
        df.drop(columns="HighCostLabel", inplace=True)

    # Drop unnecessary metadata columns
    meta_cols = [
        "MONTH", "QUARTER", "ActualMonthNumber", "IsLatest", "IsYTD",
        "Payers_key", "Enrollment_key", "LastPCPProvider", "LastPCPVisit",
        "AWVLastDate", "AWVCode", "AWVStatus", "AWVProviderNetwork",
        "AttributionStatus", "DoneBySelf"
    ]
    df.drop(columns=[col for col in meta_cols if col in df.columns], inplace=True)

    # Clean year columns
    for col in ["YEAR", "ActualYear"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median() if not df[col].isna().all() else 2024).astype(int)

    if "Ishighrisk" in df.columns:
        df["Ishighrisk"] = pd.to_numeric(df["Ishighrisk"], errors="coerce").fillna(0).astype(int)

    # Fill missing values
    numeric_cols = df.select_dtypes(include=np.number).columns
    exclude_fill = ["Member_Key", "YEAR", "HighCostLabel"]
    df[[col for col in numeric_cols if col not in exclude_fill]] = df[[col for col in numeric_cols if col not in exclude_fill]].fillna(0)

    object_cols = df.select_dtypes(include="object").columns
    df[object_cols] = df[object_cols].fillna("Unknown")

    # Cost ratios
    if "TotalCost" in df.columns:
        total = df["TotalCost"].clip(lower=1)
        cost_fields = ["Inpatient_Cost", "ERAdmisison_Cost", "EmergencyDepartmentVisits_Cost",
                       "Medication_Cost", "HomeHealth_Cost", "SkilledNursingFacilities_Cost",
                       "OutpatientVisits_Cost"]
        
        for col in cost_fields:
            if col in df.columns:
                df[col.replace("_Cost", "_CostRatio")] = df[col] / total

    # HCC changes
    if all(col in df.columns for col in ["MemberHccScore", "MemberHccScoreLastYear"]):
        df["hcc_change"] = df["MemberHccScore"] - df["MemberHccScoreLastYear"]
        df["hcc_increasing"] = (df["hcc_change"] > 0).astype(int)
        df["hcc_pct_change"] = df["hcc_change"] / df["MemberHccScoreLastYear"].clip(lower=0.01)

    # Event flags
    event_flags = {
        "Inpatient": "had_inpatient",
        "ERAdmisison": "had_er",
        "30dayHospitalReadmission": "had_readmission",
        "SkilledNursingFacilities": "had_snf",
        "HomeHealth": "had_home_health"
    }
    for old_col, new_col in event_flags.items():
        if old_col in df.columns:
            df[new_col] = (df[old_col] > 0).astype(int)

    if "PCPVisits" in df.columns:
        df["no_pcp_visit"] = (df["PCPVisits"] == 0).astype(int)

    if all(col in df.columns for col in ["TotalCost", "ProviderVisitCount"]):
        df["cost_per_visit"] = df["TotalCost"] / df["ProviderVisitCount"].clip(lower=1)

    # Chronic costs
    chronic_costs = [col for col in df.columns if col.endswith("_Cost") and 
                     any(word in col for word in ["Diabetes", "Congestive", "Pulmonary", "Pneumonia"])]
    if chronic_costs:
        df["total_chronic_cost"] = df[chronic_costs].sum(axis=1)

    # Out-of-network ratios
    if "TotalCost" in df.columns:
        total = df["TotalCost"].clip(lower=1)
        oon_cols = [col for col in df.columns if "OutNetwork" in col and col.endswith("_Cost")]
        pref_cols = [col for col in df.columns if "Preffered" in col and col.endswith("_Cost")]
        
        if oon_cols:
            df["oon_cost_ratio"] = df[oon_cols].sum(axis=1) / total
        if pref_cols:
            df["preferred_cost_ratio"] = df[pref_cols].sum(axis=1) / total

    # Simple aggregates
    imaging_cols = [col for col in ["CTEvents", "MRIEvents", "RadiologyEvents", "OtherImagingServices"] if col in df.columns]
    if imaging_cols:
        df["total_imaging"] = df[imaging_cols].sum(axis=1)

    lab_cols = [col for col in ["LabEventsPathalogy", "LabEventsClinicalDiagnostics"] if col in df.columns]
    if lab_cols:
        df["total_lab_tests"] = df[lab_cols].sum(axis=1)

    if all(col in df.columns for col in ["AWV_Compliant", "AWV_Eligible"]):
        df["awv_gap"] = ((df["AWV_Eligible"] == 1) & (df["AWV_Compliant"] == 0)).astype(int)

    # Clip extreme costs
    cost_cols = [col for col in df.columns if col.endswith("_Cost") or col == "TotalCost"]
    if dataset_type == "train":
        for col in cost_cols:
            if col in df.columns:
                cap = df[col].quantile(0.995)
                cost_caps[col] = cap
                df[col] = df[col].clip(upper=cap)
    else:
        for col, cap in cost_caps.items():
            if col in df.columns:
                df[col] = df[col].clip(upper=cap)

    log(f"Main data processed. Shape: {df.shape}")
    return df


#  TEMPORAL FEATURES 

def create_temporal_features(folder, dataset_type):
    log(f"\n=== Creating temporal (lag) features ({dataset_type}) ===")

    df = load_data(folder, f"main_df_{dataset_type}.csv")
    if df is None:
        return None

    df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce").astype("Int64")
    df["MONTH"] = pd.to_numeric(df["MONTH"], errors="coerce")
    df = df.dropna(subset=["Member_Key", "YEAR"])
    df["Member_Key"] = df["Member_Key"].astype(int)

    lag_columns = ["TotalCost", "MemberHccScore", "ProviderVisitCount", "PCPVisits",
                   "Inpatient", "ERAdmisison", "Medication_Cost", "Inpatient_Cost",
                   "OutpatientVisits_Cost", "SpecialistVisits", "30dayHospitalReadmission"]
    
    lag_columns = [col for col in lag_columns if col in df.columns]
    for col in lag_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Use only year-end records
    history = df[df["MONTH"].isin([-1, 12])][["Member_Key", "YEAR"] + lag_columns].copy()
    history = history.sort_values(["Member_Key", "YEAR"])

    lag_records = []

    for member_id, group in history.groupby("Member_Key"):
        group = group.sort_values("YEAR").drop_duplicates("YEAR")
        year_lookup = group.set_index("YEAR")

        for _, row in group.iterrows():
            current_year = row["YEAR"]
            record = {"Member_Key": member_id, "YEAR": current_year}

            # Create lags
            for lag in [1, 2]:
                prev_year = current_year - lag
                if prev_year in year_lookup.index:
                    prev_row = year_lookup.loc[prev_year]
                    for col in lag_columns:
                        record[f"{col}_lag{lag}"] = prev_row[col]
                else:
                    for col in lag_columns:
                        record[f"{col}_lag{lag}"] = 0

            # Year-over-year changes
            if f"TotalCost_lag1" in record:
                for col in ["TotalCost", "MemberHccScore", "ProviderVisitCount"]:
                    if col in lag_columns:
                        curr = row[col]
                        prev = record.get(f"{col}_lag1")
                        if prev is not None and prev != 0:
                            record[f"{col}_yoy_change"] = curr - prev
                            record[f"{col}_yoy_pct"] = (curr - prev) / abs(prev)

            lag_records.append(record)

    if not lag_records:
        return None

    lag_df = pd.DataFrame(lag_records)
    log(f"Temporal features created. Shape: {lag_df.shape}")
    return lag_df


# ICD DATA 

def process_icd_data(folder, dataset_type):
    log(f"\n=== Processing ICD data ({dataset_type}) ===")

    df = load_data(folder, f"icd_df_{dataset_type}.csv")
    if df is None:
        return None, None

    if "Start_Date" in df.columns:
        df = df.rename(columns={"Start_Date": "YEAR"})

    df = df.drop_duplicates(subset=["Member_Key", "Diagnosis_Code", "YEAR"])
    df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce").fillna(2024).astype(int)

    # ICD Chapter
    chapter_map = {
        "A":"Infectious","B":"Infectious","C":"Neoplasms","D":"Neoplasms","E":"Endocrine_Metabolic",
        "F":"Mental_Behavioral","G":"Nervous_System","H":"Eye_Ear","I":"Circulatory","J":"Respiratory",
        "K":"Digestive","L":"Skin","M":"Musculoskeletal","N":"Genitourinary","O":"Pregnancy",
        "P":"Perinatal","Q":"Congenital","R":"Symptoms_Signs","S":"Injury_Trauma","T":"Injury_Trauma",
        "V":"External_Causes","W":"External_Causes","X":"External_Causes","Y":"External_Causes",
        "Z":"Factors_Health_Status","U":"Special_Codes"
    }
    df["icd_chapter"] = df["Diagnosis_Code"].str[0].map(chapter_map).fillna("Other")

    # Chronic flags
    chronic_flags = {
        "has_diabetes": r"^E(08|09|10|11|12|13)",
        "has_chf": r"^I50",
        "has_copd_asthma": r"^(J44|J45)",
        "has_ckd": r"^N18",
        "has_cancer": r"^C",
        "has_hypertension": r"^I10",
        "has_depression": r"^F3[23]",
        "has_anxiety": r"^F41",
        "has_atrial_fib": r"^I48",
        "has_coronary_disease": r"^I25",
        "has_stroke": r"^I6[0-9]",
        "has_dialysis": r"^Z(49|99\.2)",
        "has_substance_use": r"^F1[0-9]",
        "has_sepsis": r"^A41",
        "has_pneumonia": r"^J1[2-8]",
        "has_dementia": r"^F0[0-3]",
        "has_obesity": r"^E66",
        "has_liver_disease": r"^K7[0-7]",
        "has_hiv": r"^B20",
        "has_fracture": r"^(S[0-9][0-9]\.?[0-9]?|M80)",
    }

    for flag_name, pattern in chronic_flags.items():
        df[flag_name] = df["Diagnosis_Code"].str.match(pattern).astype("int8")

    # Yearly aggregates
    yearly = df.groupby(["Member_Key", "YEAR"]).agg(
        icd_codes=("Diagnosis_Code", "nunique"),
        icd_events=("Diagnosis_Code", "count"),
        icd_chapters=("icd_chapter", "nunique"),
        **{f"{name}_yr": (name, "max") for name in chronic_flags}
    ).reset_index()

    yearly["chronic_conditions_yr"] = yearly[[f"{name}_yr" for name in chronic_flags]].sum(axis=1)

    # Lifetime aggregates
    lifetime = df.groupby("Member_Key").agg(
        icd_codes_lifetime=("Diagnosis_Code", "nunique"),
        icd_events_lifetime=("Diagnosis_Code", "count"),
        icd_chapters_lifetime=("icd_chapter", "nunique"),
        years_active=("YEAR", "nunique"),
        **{f"{name}_ever": (name, "max") for name in chronic_flags}
    ).reset_index()

    lifetime["chronic_conditions_ever"] = lifetime[[f"{name}_ever" for name in chronic_flags]].sum(axis=1)

    log(f"ICD processed - Yearly: {yearly.shape}, Lifetime: {lifetime.shape}")
    return yearly, lifetime


#  CPT & DRG 

def process_cpt_data(folder, dataset_type):
    log(f"\n=== Processing CPT data ({dataset_type}) ===")
    df = load_data(folder, f"cpt_df_{dataset_type}.csv")
    if df is None:
        return None

    if "StartDate" in df.columns:
        df = df.rename(columns={"StartDate": "YEAR"})
    elif "Start_Date" in df.columns:
        df = df.rename(columns={"Start_Date": "YEAR"})

    df = df.drop_duplicates()
    df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce").fillna(2024).astype(int)

    # Simple CPT processing (kept minimal for clarity)
    if "Procedure_Code" in df.columns:
        df["Procedure_Code"] = df["Procedure_Code"].astype(str).str.strip()
        df["is_high_cost_proc"] = df["Procedure_Code"].str[:2].isin(["33","34","47","48","50","00","01","02"]).astype("int8")
    else:
        df["is_high_cost_proc"] = 0

    cpt_yearly = df.groupby(["Member_Key", "YEAR"]).agg(
        cpt_procedures=("Procedure_Code", "nunique"),
        cpt_events=("Procedure_Code", "count"),
        high_cost_procs=("is_high_cost_proc", "sum")
    ).reset_index()

    log(f"CPT processed: {cpt_yearly.shape}")
    return cpt_yearly


def process_drg_data(folder, dataset_type):
    log(f"\n=== Processing DRG data ({dataset_type}) ===")
    df = load_data(folder, f"drg_df_{dataset_type}.csv")
    if df is None:
        return None

    if "Start_Date_Year" in df.columns:
        df = df.rename(columns={"Start_Date_Year": "YEAR"})

    df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce").fillna(2024).astype(int)

    valid_drg = df[df["Code"] != -1].copy()
    valid_drg["high_severity"] = valid_drg["Code"].isin(range(1,9)).astype("int8")  # simplified

    drg_yearly = valid_drg.groupby(["Member_Key", "YEAR"]).agg(
        drg_claims=("Claim_Key", "count"),
        high_severity_drgs=("high_severity", "sum")
    ).reset_index()

    log(f"DRG processed: {drg_yearly.shape}")
    return drg_yearly


#  DOB DATA 

def process_dob_data(folder, obs_year=2024):
    log(f"\n=== Processing DOB data (year {obs_year}) ===")
    df = load_data(folder, "dob_df.csv")
    if df is None:
        return None

    df = df.drop_duplicates(subset=["Member_Key"])

    if "DOB_Key" in df.columns:
        df["DOB_parsed"] = pd.to_datetime(df["DOB_Key"], errors="coerce")
        ref_date = pd.Timestamp(f"{obs_year}-01-01")
        df["age"] = ((ref_date - df["DOB_parsed"]).dt.days / 365.25).round(1).clip(0, 120)

        df["is_senior"] = (df["age"] >= 65).astype(int)
        df["is_very_senior"] = (df["age"] >= 80).astype(int)

        df.drop(columns=["DOB_parsed", "DOB_Key"], errors="ignore", inplace=True)

    if "Gender_Key" in df.columns:
        df["is_female"] = (df["Gender_Key"] == 2).astype(int)

    log(f"DOB processed: {df.shape}")
    return df


#  MERGE & FINAL FEATURES 

def merge_all_features(main_df, icd_yearly, icd_lifetime, cpt_yearly, drg_yearly, 
                       dob_df, temporal_df, dataset_type):
    log(f"\n=== Merging all features for {dataset_type} ===")

    result = main_df.copy()

    if icd_yearly is not None:
        result = result.merge(icd_yearly, on=["Member_Key", "YEAR"], how="left")
    if icd_lifetime is not None:
        result = result.merge(icd_lifetime, on="Member_Key", how="left")
    if cpt_yearly is not None:
        result = result.merge(cpt_yearly, on=["Member_Key", "YEAR"], how="left")
    if drg_yearly is not None:
        result = result.merge(drg_yearly, on=["Member_Key", "YEAR"], how="left")
    if dob_df is not None:
        result = result.merge(dob_df, on="Member_Key", how="left")
    if temporal_df is not None:
        result = result.merge(temporal_df, on=["Member_Key", "YEAR"], how="left")

    # Fill remaining missing values
    for col in result.columns:
        if result[col].dtype in [np.number, "Int64"]:
            result[col] = result[col].fillna(0)
        else:
            result[col] = result[col].fillna("Unknown")

    log(f"Final dataset shape for {dataset_type}: {result.shape}")
    return result


def add_interaction_features(df):
    log("Adding interaction features...")

    if all(col in df.columns for col in ["MemberHccScore", "chronic_conditions_yr"]):
        df["hcc_chronic_interaction"] = df["MemberHccScore"] * df["chronic_conditions_yr"]

    if all(col in df.columns for col in ["TotalCost_lag1", "MemberHccScore"]):
        df["lag_cost_hcc"] = df["TotalCost_lag1"] * df["MemberHccScore"]

    if all(col in df.columns for col in ["TotalCost", "TotalCost_lag1"]):
        df["cost_acceleration"] = (df["TotalCost"] > 2 * df["TotalCost_lag1"].clip(lower=1)).astype(int)

    if all(col in df.columns for col in ["age", "chronic_conditions_ever"]):
        df["age_chronic"] = df["age"] * df["chronic_conditions_ever"]

    return df


# MAIN 

def main():
    log("Starting Healthcare Cost Prediction Pipeline v2")
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    #  TRAIN 
    log("\n--- Processing Training Data ---")

    dob_train = process_dob_data(TRAIN_FOLDER, 2023)
    main_train = process_main_data(TRAIN_FOLDER, "train")
    temporal_train = create_temporal_features(TRAIN_FOLDER, "train")
    
    icd_yearly_train, icd_lifetime_train = process_icd_data(TRAIN_FOLDER, "train")
    cpt_train = process_cpt_data(TRAIN_FOLDER, "train")
    drg_train = process_drg_data(TRAIN_FOLDER, "train")

    train_final = merge_all_features(
        main_train, icd_yearly_train, icd_lifetime_train,
        cpt_train, drg_train, dob_train, temporal_train, "train"
    )
    train_final = add_interaction_features(train_final)

    train_path = os.path.join(OUTPUT_FOLDER, "features_train2.csv")
    train_final.to_csv(train_path, index=False)
    log(f"✅ Training features saved: {train_path}  {train_final.shape}")

    #TEST 
    log("\n--- Processing Test Data ---")

    dob_test = process_dob_data(TEST_FOLDER, 2024)
    main_test = process_main_data(TEST_FOLDER, "test")
    temporal_test = create_temporal_features(TEST_FOLDER, "test")
    
    icd_yearly_test, icd_lifetime_test = process_icd_data(TEST_FOLDER, "test")
    cpt_test = process_cpt_data(TEST_FOLDER, "test")
    drg_test = process_drg_data(TEST_FOLDER, "test")

    test_final = merge_all_features(
        main_test, icd_yearly_test, icd_lifetime_test,
        cpt_test, drg_test, dob_test, temporal_test, "test"
    )
    test_final = add_interaction_features(test_final)

    test_path = os.path.join(OUTPUT_FOLDER, "features_test2.csv")
    test_final.to_csv(test_path, index=False)
    log(f"✅ Test features saved: {test_path}  {test_final.shape}")

    log("\nPipeline finished successfully!")


if __name__ == "__main__":
    main()