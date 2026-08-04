#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import re
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================

# Source Roots
DROPBOX_ROOT = "/Users/zungrulin/Library/CloudStorage/Dropbox/ML for Peace"
CIVIC_SRC_ROOT = os.path.join(DROPBOX_ROOT, "Counts_Civic_New", "Final_Aggregated")
RAI_SRC_ROOT   = os.path.join(DROPBOX_ROOT, "Counts_RAI_New",   "Final_Aggregated")
MLEED_SRC_ROOT = os.path.join(DROPBOX_ROOT, "Counts_Env",       "Final_Aggregated")  # NEW

# Destination Roots (repo)
REPO_ROOT     = str(Path(__file__).resolve().parents[1] / "data")
CIVIC_DEST_DIR = os.path.join(REPO_ROOT, "1-civic-aggregate")
RAI_DEST_DIR   = os.path.join(REPO_ROOT, "1-rai-aggregate")
MLEED_DEST_DIR = os.path.join(REPO_ROOT, "1-mleed-aggregate")  # NEW (rename to 1-env-aggregate if you prefer)

# Ensure destination directories exist
os.makedirs(CIVIC_DEST_DIR, exist_ok=True)
os.makedirs(RAI_DEST_DIR,   exist_ok=True)
os.makedirs(MLEED_DEST_DIR, exist_ok=True)  # NEW

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def get_latest_date_folder(country_path):
    """Find latest YYYY_M_D subfolder."""
    if not os.path.exists(country_path):
        return None
    subdirs = [d for d in os.listdir(country_path) if os.path.isdir(os.path.join(country_path, d))]
    date_dirs = [d for d in subdirs if re.fullmatch(r'\d{4}_\d{1,2}_\d{1,2}', d)]
    if not date_dirs:
        return None
    def date_key(name):
        year, month, day = name.split('_')
        return int(year), int(month), int(day)
    return max(date_dirs, key=date_key)

def normalize_date(df):
    """Converts 'date' to first of month (YYYY-MM-01) if present."""
    if 'date' in df.columns:
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['date'] = df['date'].dt.to_period('M').dt.to_timestamp()  # YYYY-MM-01
    return df

def rename_norm_columns(df, prefix=""):
    """
    Renames xxx_norm -> xxxNorm; if prefix provided (e.g., 'ncr_'), adds it to non-key columns.
    Keys (date/year/month/country/influencer) are kept unchanged.
    """
    df = df.copy()
    keys = {'date', 'year', 'month', 'country', 'influencer'}
    new_cols = {}
    for col in df.columns:
        if col in keys:
            new_cols[col] = col
            continue
        new_name = col
        if new_name.endswith('_norm'):
            new_name = new_name[:-5] + 'Norm'
        if prefix:
            new_name = prefix + new_name
        new_cols[col] = new_name
    return df.rename(columns=new_cols)

# ==========================================
# CIVIC DATA PROCESSING (UPDATED FOR COMBINED/CR/NR)
# ==========================================
print(f"Processing Civic Data from: {CIVIC_SRC_ROOT}")
if os.path.exists(CIVIC_SRC_ROOT):
    countries = [d for d in os.listdir(CIVIC_SRC_ROOT) if os.path.isdir(os.path.join(CIVIC_SRC_ROOT, d))]
else:
    countries = []

for country in countries:
    country_path = os.path.join(CIVIC_SRC_ROOT, country)
    date_folder = get_latest_date_folder(country_path)
    if not date_folder:
        print(f"[CIVIC] Skipping {country}: No date folder found.")
        continue

    full_path = os.path.join(country_path, date_folder)
    file_cr  = os.path.join(full_path, f"{country}_Civic_Related.csv")
    file_ncr = os.path.join(full_path, f"{country}_Non_Civic_Related.csv")

    if os.path.exists(file_cr) and os.path.exists(file_ncr):
        try:
            df_cr  = pd.read_csv(file_cr)
            df_ncr = pd.read_csv(file_ncr)

            # Normalize dates
            df_cr  = normalize_date(df_cr)
            df_ncr = normalize_date(df_ncr)

            # Standardize naming: xxx_norm -> xxxNorm, no prefixes yet
            df_cr  = rename_norm_columns(df_cr, prefix="")
            df_ncr = rename_norm_columns(df_ncr, prefix="")

            # Merge CR and NCR on 'date' (and year/month if present)
            merge_keys = [k for k in ['date', 'year', 'month'] if k in df_cr.columns and k in df_ncr.columns]
            if 'date' not in merge_keys:
                merge_keys = ['date']

            # Restrict to keys + event count / normalized columns
            def raw_event_cols(df):
                keys = {'date', 'year', 'month', 'country', 'influencer', 'total_articles'}
                return [c for c in df.columns if c not in keys and not c.endswith('Norm')]

            def norm_cols(df):
                return [c for c in df.columns
                        if c.endswith('Norm') and c not in ['country', 'influencer']]

            cr_raws = raw_event_cols(df_cr)
            ncr_raws = raw_event_cols(df_ncr)
            cr_norms  = norm_cols(df_cr)
            ncr_norms = norm_cols(df_ncr)

            # Align columns: assume same event set, but be robust
            all_raw_events = sorted(set(cr_raws) | set(ncr_raws))
            all_norm_events = sorted(set(cr_norms) | set(ncr_norms))

            # Fill missing Norm columns with 0 so we can sum safely
            for c in all_raw_events:
                if c not in df_cr.columns:
                    df_cr[c] = 0
                if c not in df_ncr.columns:
                    df_ncr[c] = 0
            for c in all_norm_events:
                if c not in df_cr.columns:
                    df_cr[c] = 0.0
                if c not in df_ncr.columns:
                    df_ncr[c] = 0.0

            cr_select  = merge_keys + all_raw_events + all_norm_events
            ncr_select = merge_keys + all_raw_events + all_norm_events

            df_cr_sel  = df_cr[cr_select].copy()
            df_ncr_sel = df_ncr[ncr_select].copy()

            merged = pd.merge(df_cr_sel, df_ncr_sel,
                              on=merge_keys,
                              suffixes=("_cr", "_ncr"),
                              how="outer")

            # Build output:
            # - combined event columns: eventNorm = CR + NCR
            # - civic-related: cr_eventNorm = CR only
            # - non-civic:    nr_eventNorm = NCR only
            out = merged[merge_keys].copy()

            for ev in all_raw_events:
                cr_col  = f"{ev}_cr"
                ncr_col = f"{ev}_ncr"

                if cr_col not in merged.columns:
                    merged[cr_col] = 0
                if ncr_col not in merged.columns:
                    merged[ncr_col] = 0

                out[ev] = merged[cr_col].fillna(0).astype(int) + merged[ncr_col].fillna(0).astype(int)
                out[f"cr_{ev}"] = merged[cr_col].fillna(0).astype(int)
                out[f"nr_{ev}"] = merged[ncr_col].fillna(0).astype(int)

            for ev in all_norm_events:
                cr_col  = f"{ev}_cr"
                ncr_col = f"{ev}_ncr"

                # ensure columns exist in merged
                if cr_col not in merged.columns:
                    merged[cr_col] = 0.0
                if ncr_col not in merged.columns:
                    merged[ncr_col] = 0.0

                # Combined
                out[ev] = merged[cr_col].astype(float) + merged[ncr_col].astype(float)
                # CR-only
                out[f"cr_{ev}"] = merged[cr_col].astype(float)
                # NR-only (your request says 'nr_' not 'ncr_')
                out[f"nr_{ev}"] = merged[ncr_col].astype(float)

            # Optional: keep total_articles if present in CR file
            if 'total_articles' in df_cr.columns:
                out['total_articles'] = df_cr['total_articles']

            # Save
            out_file = os.path.join(CIVIC_DEST_DIR, f"{country}.csv")
            out.to_csv(out_file, index=False)
            print(f"[CIVIC] Saved: {out_file}")
        except Exception as e:
            print(f"[CIVIC] Error processing {country}: {e}")
    else:
        print(f"[CIVIC] Missing files for {country} in {full_path}")


# ==========================================
# RAI DATA PROCESSING (unchanged)
# ==========================================
print(f"\nProcessing RAI Data from: {RAI_SRC_ROOT}")
if os.path.exists(RAI_SRC_ROOT):
    countries_rai = [d for d in os.listdir(RAI_SRC_ROOT) if os.path.isdir(os.path.join(RAI_SRC_ROOT, d))]
else:
    countries_rai = []

for country in countries_rai:
    country_path = os.path.join(RAI_SRC_ROOT, country)
    date_folder = get_latest_date_folder(country_path)

    if not date_folder:
        print(f"[RAI] Skipping {country}: No date folder found.")
        continue

    full_path = os.path.join(country_path, date_folder)

    influencers = {
        'China':    f"{country}_China.csv",
        'Russia':   f"{country}_Russia.csv",
        'Combined': f"{country}_Combined.csv"
    }

    for inf_name, filename in influencers.items():
        file_path = os.path.join(full_path, filename)
        if not os.path.exists(file_path):
            print(f"[RAI] Missing file for {country} {inf_name}: {file_path}")
            continue

        try:
            df = pd.read_csv(file_path)
            df = normalize_date(df)

            # Add metadata
            df['country'] = country
            df['influencer'] = inf_name

            # Reorder columns
            cols = df.columns.tolist()
            front = ['country', 'influencer', 'date']
            other = [c for c in cols if c not in front]
            df = df[front + other]

            # Rename xxx_norm -> xxxNorm
            df = rename_norm_columns(df)

            out_filename = f"{country}_{inf_name.lower()}.csv"
            out_file = os.path.join(RAI_DEST_DIR, out_filename)
            df.to_csv(out_file, index=False)
            print(f"[RAI] Saved: {out_file}")
        except Exception as e:
            print(f"[RAI] Error processing {country} {inf_name}: {e}")

# ==========================================
# MLEED (ENV) DATA PROCESSING  <-- NEW
# ==========================================
print(f"\nProcessing MLEED (Env) Data from: {MLEED_SRC_ROOT}")
if os.path.exists(MLEED_SRC_ROOT):
    countries_env = [d for d in os.listdir(MLEED_SRC_ROOT) if os.path.isdir(os.path.join(MLEED_SRC_ROOT, d))]
else:
    countries_env = []

for country in countries_env:
    country_path = os.path.join(MLEED_SRC_ROOT, country)
    date_folder = get_latest_date_folder(country_path)

    if not date_folder:
        print(f"[MLEED] Skipping {country}: No date folder found.")
        continue

    full_path = os.path.join(country_path, date_folder)
    file_path = os.path.join(full_path, f"{country}.csv")  # single flat file per country

    if not os.path.exists(file_path):
        print(f"[MLEED] Missing file for {country}: {file_path}")
        continue

    try:
        df = pd.read_csv(file_path)
        df = normalize_date(df)
        # Standardize xxx_norm -> xxxNorm (consistent with civic/rai)
        df = rename_norm_columns(df)

        out_file = os.path.join(MLEED_DEST_DIR, f"{country}.csv")
        df.to_csv(out_file, index=False)
        print(f"[MLEED] Saved: {out_file}")
    except Exception as e:
        print(f"[MLEED] Error processing {country}: {e}")
