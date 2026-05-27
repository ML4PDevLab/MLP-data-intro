 
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn import tree as tr
import data
from datetime import datetime,date
import warnings
warnings.filterwarnings('ignore')
from peak_detector import PeakDetector
import matplotlib.dates as mdates

# =========================
# Global Variables Inputs
# =========================
today = date.today()
day = today.strftime("%d")
month = today.strftime("%m")
year = today.strftime("%Y")

# =========================
# Paths (relative)
# =========================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
USE_TF_PEAK_MODEL = os.environ.get("MLP_USE_TF", "0") == "1"

civic_data_folder = os.path.join(SCRIPT_DIR, '../../data/1-civic-aggregate')
rai_data_folder   = os.path.join(SCRIPT_DIR, '../../data/1-rai-aggregate')
mleed_data_folder = os.path.join(SCRIPT_DIR, '../../data/1-mleed-aggregate')   # <-- NEW

civic_result_folder = os.path.join(SCRIPT_DIR, '../../data/2-civic-shock')
rai_result_folder   = os.path.join(SCRIPT_DIR, '../../data/2-rai-shock')
mleed_result_folder = os.path.join(SCRIPT_DIR, '../../data/2-mleed-shock')     # <-- NEW

os.makedirs(civic_result_folder, exist_ok=True)
os.makedirs(rai_result_folder,   exist_ok=True)
os.makedirs(mleed_result_folder, exist_ok=True)  # <-- NEW

# =========================
# Countries
# =========================
country_list = [

                # 'Guatemala', 'Honduras',  'El Salvador', 'Nicaragua'
    
                'Albania', 'Algeria', 'Angola', 'Armenia', 'Azerbaijan', 'Bangladesh', 'Belarus', 'Benin', 'Burkina Faso',
                'Cambodia', 'Cameroon', 'Colombia', 'Costa Rica', 'DR Congo', 'Dominican Republic', 'Ecuador', 'El Salvador',
                'Ethiopia', 'Georgia', 'Ghana', 'Guatemala', 'Honduras', 'Hungary', 'India', 'Indonesia', 'Jamaica', 'Kazakhstan',
                'Kenya', 'Kosovo', 'Kyrgyzstan', 'Liberia', 'Macedonia', 'Malawi', 'Malaysia', 'Mali', 'Mauritania', 'Mexico', 'Moldova',
                'Morocco', 'Mozambique', 'Namibia', 'Nepal', 'Nicaragua', 'Niger', 'Nigeria', 'Pakistan', 'Panama', 'Paraguay',
                'Peru', 'Philippines', 'Rwanda', 'Senegal', 'Serbia', 'Solomon Islands', 'South Africa', 'South Sudan', 'Sri Lanka',
                'Tanzania', 'Timor Leste', 'Tunisia', 'Turkey', 'Uganda', 'Ukraine', 'Uzbekistan', 'Zambia', 'Zimbabwe'
                ]

# =========================
# Helpers
# =========================
def get_updated_files(path='.'):
    if not os.path.exists(path):
        return None, []
    files = os.listdir(path)
    csv_files = [f for f in files if f.endswith('.csv')]
    remove_files = ['.ipynb_checkpoints', 'full-data.csv', 'full-data.rds']
    for file in remove_files:
        if file in csv_files:
            csv_files.remove(file)
    return path + '/', csv_files

def _load_tfsm_model():
    """Load TF SavedModel for peak fusion (same as your civic/rai path)."""
    if not USE_TF_PEAK_MODEL:
        return None

    from keras import Input, Model
    import tensorflow as tf

    script_dir = os.path.dirname(__file__)
    model_path = os.path.join(script_dir, 'content', 'model_version1')
    TFSMLayer = tf.keras.layers.TFSMLayer
    layer = TFSMLayer(model_path, call_endpoint='serving_default')
    inp = Input(shape=(2,))
    out = layer(inp)
    return Model(inputs=inp, outputs=out)

def convert_to_training_data_2(Y, country, event, peak_detector, loaded_model=None):
    X_values = peak_detector.peak_detection(Y)
    X_values = np.nan_to_num(np.array(X_values))
    X_values = X_values.reshape(-1, 2)

    if loaded_model is None:
        binary_predictions_NN = [0] * len(X_values)
    else:
        predictions = loaded_model.predict(X_values)
        predictions = predictions['dense_5']
        binary_predictions_NN = (predictions > 0.5).astype(int)
        binary_predictions_NN = [item for sublist in binary_predictions_NN for item in sublist]

    binary_predictions_algorithm = peak_detector.peak_detection_conservative(Y)
    binary_predictions_algorithm = [int(v) for v in binary_predictions_algorithm]

    # OR of NN + conservative algorithm
    new_list = [1 if x or y else 0 for x, y in zip(binary_predictions_NN, binary_predictions_algorithm)]

    # force top-3 magnitude spikes on
    top_3_indices = sorted(range(len(Y)), key=lambda i: abs(Y[i]), reverse=True)[:3]
    for idx in top_3_indices:
        new_list[idx] = 1

    # neighborhood cleanup: drop pits & allow neighbor switch
    for i in range(1, len(Y) - 1):
        if new_list[i] == 1:
            left = Y[i-1] if i > 0 else float('-inf')
            right = Y[i+1] if i < len(Y) - 1 else float('-inf')
            if left > Y[i] and right > Y[i]:
                new_list[i] = 0
            else:
                if left > Y[i]:
                    new_list[i-1] = 1
                if right > Y[i]:
                    new_list[i+1] = 1

    return X_values, binary_predictions_algorithm, new_list

# =========================
# CIVIC peak detection
# =========================
def detect_peaks(folder, countries, date):
    civic_plot_dir = civic_result_folder
    os.makedirs(civic_plot_dir, exist_ok=True)

    lookaround = 12
    std_dev = 0.88
    normalise = 1
    alpha = 0.05
    beta = 0.2
    peak_detector = PeakDetector(lookaround, std_dev, normalise, alpha, beta)
    loaded_model = _load_tfsm_model()

    peak_results = {}
    for country in country_list:
        if country in countries:
            file = country + '.csv'
            path = os.path.join(folder, file)
            if not os.path.exists(path):
                continue

            civic_data = pd.read_csv(path)

            # ---- figure out which columns to run shocks on ----
            # 1) combined series: eventNorm (from data.civic)
            combined_events = [e for e in data.civic if e in civic_data.columns]

            # 2) civic‑related series: cr_eventNorm if present
            cr_events = [f"cr_{e}" for e in data.civic if f"cr_{e}" in civic_data.columns]

            # all event columns we’ll create shock flags for
            all_events = combined_events + cr_events

            if not all_events:
                print(f"[CIVIC] {country}: no matching combined/CR columns found; skipped.")
                continue

            # build peaks_df with date + all event columns
            cols = ['date'] + all_events
            peaks_df = pd.DataFrame(columns=cols)
            peaks_df['date'] = civic_data['date'].tolist()

            # ---------- run shocks for combined series ----------
            for event in combined_events:
                Y = civic_data[event]
                _, _, peaks_detected = convert_to_training_data_2(
                    Y, country, event, peak_detector, loaded_model
                )
                peaks_df[event] = peaks_detected

                # plotting
                dfp = civic_data.copy()
                dfp[event + '_peaks'] = peaks_detected
                dfp['date'] = pd.to_datetime(dfp['date'])

                fig, ax2 = plt.subplots(figsize=(15, 12))
                ax2.plot(dfp['date'], dfp[event],
                         label='Normalized Number of Articles', color='green')
                ax2.scatter(
                    dfp['date'][dfp[event + '_peaks'] == 1],
                    dfp[event][dfp[event + '_peaks'] == 1],
                    color='red', label='Detected Peaks', zorder=5
                )
                ax2.set_xlabel('Date')
                ax2.set_ylabel('Normalized Number of Articles')
                ax2.legend(loc='upper right')
                ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=5))
                ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                plt.xticks(rotation=45)
                plt.title(f'Peak Detection for {country} - {event}')
                plt.savefig(os.path.join(civic_plot_dir, f'{country}_{event}_peaks.png'))
                plt.close()

            # ---------- run shocks for CR series (cr_*) ----------
            for event in cr_events:
                Y = civic_data[event]
                _, _, peaks_detected = convert_to_training_data_2(
                    Y, country, event, peak_detector, loaded_model
                )
                peaks_df[event] = peaks_detected

                dfp = civic_data.copy()
                dfp[event + '_peaks'] = peaks_detected
                dfp['date'] = pd.to_datetime(dfp['date'])

                fig, ax2 = plt.subplots(figsize=(15, 12))
                ax2.plot(dfp['date'], dfp[event],
                         label='Normalized Number of Articles', color='green')
                ax2.scatter(
                    dfp['date'][dfp[event + '_peaks'] == 1],
                    dfp[event][dfp[event + '_peaks'] == 1],
                    color='red', label='Detected Peaks', zorder=5
                )
                ax2.set_xlabel('Date')
                ax2.set_ylabel('Normalized Number of Articles')
                ax2.legend(loc='upper right')
                ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=5))
                ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                plt.xticks(rotation=45)
                plt.title(f'Peak Detection for {country} - {event}')
                plt.savefig(os.path.join(civic_plot_dir, f'{country}_{event}_peaks.png'))
                plt.close()

            # save civic peaks (combined + CR) for this country
            outfile = os.path.join(civic_result_folder, f'{country}.csv')
            peaks_df.to_csv(outfile, index=False)

    return peak_results

# =========================
# RAI peak detection
# =========================
def detect_rai_peaks_by_influencer(folder, countries, date):
    rai_plot_dir = rai_result_folder
    os.makedirs(rai_plot_dir, exist_ok=True)

    lookaround = 12
    std_dev = 0.88
    normalise = 1
    alpha = 0.05
    beta = 0.2
    peak_detector = PeakDetector(lookaround, std_dev, normalise, alpha, beta)
    loaded_model = _load_tfsm_model()

    peak_results = {}
    _, rai_files = get_updated_files(folder)

    for rai_file in rai_files:
        if '_' not in rai_file or not rai_file.endswith('.csv'):
            continue
        country, influencer = rai_file[:-4].split('_', 1)
        if country not in country_list:
            continue

        path = os.path.join(folder, rai_file)
        if not os.path.exists(path):
            continue
        rai_data = pd.read_csv(path)

        cols = data.rai[:]  # expected normalized RAI series (events/themes) already prepared
        cols.append('date')
        cols = [cols[-1]] + cols[:-1]
        peaks_df = pd.DataFrame(columns=cols)
        peaks_df['date'] = rai_data['date'].tolist()

        for event in data.rai:
            if event in rai_data.columns:
                Y = rai_data[event]
                _, _, peaks_detected = convert_to_training_data_2(Y, country, event, peak_detector, loaded_model)
                peaks_df[event] = peaks_detected

                dfp = rai_data.copy()
                dfp[event + '_peaks'] = peaks_detected
                dfp['date'] = pd.to_datetime(dfp['date'])

                fig, ax2 = plt.subplots(figsize=(15, 12))
                ax2.plot(dfp['date'], dfp[event], label='Normalized Number of Articles', color='green')
                ax2.scatter(dfp['date'][dfp[event + '_peaks'] == 1],
                            dfp[event][dfp[event + '_peaks'] == 1],
                            color='red', label='Detected Peaks', zorder=5)
                ax2.set_xlabel('Date')
                ax2.set_ylabel('Normalized Number of Articles')
                ax2.legend(loc='upper right')
                ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=5))
                ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                plt.xticks(rotation=45)
                plt.title(f'RAI Peak Detection for {country} ({influencer}) - {event}')
                plt.savefig(os.path.join(rai_plot_dir, f'{country}_{influencer}_{event}_peaks.png'))
                plt.close()

        outfile = os.path.join(rai_result_folder, f'{country}_{influencer}.csv')
        peaks_df.to_csv(outfile, index=False)

    return peak_results

# =========================
# MLEED peak detection  <-- NEW
# =========================
def detect_mleed_peaks(folder, countries, date):
    """
    Same logic as civic/rai, but dynamically selects MLEED normalized series
    as all columns that end with 'Norm' in each {Country}.csv under 1-mleed-aggregate.
    """
    mleed_plot_dir = mleed_result_folder
    os.makedirs(mleed_plot_dir, exist_ok=True)

    lookaround = 12
    std_dev = 0.88
    normalise = 1
    alpha = 0.05
    beta = 0.2
    peak_detector = PeakDetector(lookaround, std_dev, normalise, alpha, beta)
    loaded_model = _load_tfsm_model()

    _, csv_files = get_updated_files(folder)
    for fname in csv_files:
        # {Country}.csv only
        if '_' in fname or not fname.endswith('.csv'):
            continue
        country = fname[:-4]
        if country not in country_list:
            continue

        path = os.path.join(folder, fname)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)

        if 'date' not in df.columns:
            continue

        # Pick MLEED normalized columns in this file
        mleed_cols = [c for c in df.columns if c.endswith('Norm')]
        if not mleed_cols:
            print(f"[MLEED] {country}: no *Norm columns found; skipped.")
            continue

        cols = ['date'] + mleed_cols
        peaks_df = pd.DataFrame(columns=cols)
        peaks_df['date'] = df['date'].tolist()

        for event in mleed_cols:
            Y = df[event].values
            _, _, peaks_detected = convert_to_training_data_2(Y, country, event, peak_detector, loaded_model)
            peaks_df[event] = peaks_detected

            # Plot
            dfp = df.copy()
            dfp[event + '_peaks'] = peaks_detected
            dfp['date'] = pd.to_datetime(dfp['date'])

            fig, ax2 = plt.subplots(figsize=(15, 12))
            ax2.plot(dfp['date'], dfp[event], label='Normalized Number of Articles', color='green')
            ax2.scatter(dfp['date'][dfp[event + '_peaks'] == 1],
                        dfp[event][dfp[event + '_peaks'] == 1],
                        color='red', label='Detected Peaks', zorder=5)
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Normalized Number of Articles')
            ax2.legend(loc='upper right')
            ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=5))
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
            plt.title(f'MLEED Peak Detection for {country} - {event}')
            plt.savefig(os.path.join(mleed_plot_dir, f'{country}_{event}_peaks.png'))
            plt.close()

        outfile = os.path.join(mleed_result_folder, f'{country}.csv')
        peaks_df.to_csv(outfile, index=False)

# =========================
# Runners
# =========================
def run_peak_detection(path):
    folder, files = get_updated_files(path)
    if folder is None or len(files) == 0:
        print("No recent civic data files found.")
        return
    countries = [file[:-4] for file in files]
    detect_peaks(folder, countries, f'{year}-{month}-{day}')
    print("Civic peak detection completed.")

def run_rai_peak_detection(path):
    folder, files = get_updated_files(path)
    if folder is None or len(files) == 0:
        print("No recent RAI data files found.")
        return
    detect_rai_peaks_by_influencer(folder, [], f'{year}-{month}-{day}')
    print("RAI peak detection completed.")

def run_mleed_peak_detection(path):  # <-- NEW
    folder, files = get_updated_files(path)
    if folder is None or len(files) == 0:
        print("No recent MLEED data files found.")
        return
    # countries derive from filenames; detection function picks normalized series dynamically
    detect_mleed_peaks(folder, [], f'{year}-{month}-{day}')
    print("MLEED peak detection completed.")

# =========================
# Run peak detection
# =========================
RUN_PEAK_DETECTION = True

if RUN_PEAK_DETECTION:
    run_peak_detection(civic_data_folder)
    run_rai_peak_detection(rai_data_folder)
    run_mleed_peak_detection(mleed_data_folder)  # <-- NEW
else:
    print("⚠️  Skipping peak‑detection step (RUN_PEAK_DETECTION = False)")

# =========================
# Combine outputs
# =========================
def combine_civic_shock_files():
    civic_files = []
    for filename in os.listdir(civic_result_folder):
        if filename.endswith('.csv'):
            file_path = os.path.join(civic_result_folder, filename)
            country = filename[:-4]
            df = pd.read_csv(file_path)
            df['country'] = country
            civic_files.append(df)
    if civic_files:
        combined_civic = pd.concat(civic_files, ignore_index=True)
        cols = ['country', 'date'] + [c for c in combined_civic.columns if c not in ['country', 'date']]
        combined_civic = combined_civic[cols]
        final_counts_folder = os.path.join(SCRIPT_DIR, '../../data/final-counts')
        os.makedirs(final_counts_folder, exist_ok=True)
        output_path = os.path.join(final_counts_folder, 'shock-civic-data.csv')
        combined_civic.to_csv(output_path, index=False)
        print(f"Combined civic shock data saved to: {output_path}")
    else:
        print("No civic shock CSV files found to combine.")

def combine_rai_shock_files():
    rai_files = []
    for filename in os.listdir(rai_result_folder):
        if filename.endswith('.csv') and '_' in filename:
            file_path = os.path.join(rai_result_folder, filename)
            country, influencer = filename[:-4].split('_', 1)
            df = pd.read_csv(file_path)
            df['country'] = country
            df['influencer'] = influencer
            rai_files.append(df)
    if rai_files:
        combined_rai = pd.concat(rai_files, ignore_index=True)
        cols = ['country', 'influencer', 'date'] + [c for c in combined_rai.columns if c not in ['country', 'influencer', 'date']]
        combined_rai = combined_rai[cols]
        final_counts_folder = os.path.join(SCRIPT_DIR, '../../data/final-counts')
        os.makedirs(final_counts_folder, exist_ok=True)
        output_path = os.path.join(final_counts_folder, 'shock-rai-data.csv')
        combined_rai.to_csv(output_path, index=False)
        print(f"Combined RAI shock data saved to: {output_path}")
    else:
        print("No RAI shock CSV files found to combine.")

def combine_mleed_shock_files():  # <-- NEW
    mleed_files = []
    for filename in os.listdir(mleed_result_folder):
        if filename.endswith('.csv') and '_' not in filename:
            file_path = os.path.join(mleed_result_folder, filename)
            country = filename[:-4]
            df = pd.read_csv(file_path)
            df['country'] = country
            mleed_files.append(df)
    if mleed_files:
        combined_mleed = pd.concat(mleed_files, ignore_index=True)
        cols = ['country', 'date'] + [c for c in combined_mleed.columns if c not in ['country', 'date']]
        combined_mleed = combined_mleed[cols]
        final_counts_folder = os.path.join(SCRIPT_DIR, '../../data/final-counts')
        os.makedirs(final_counts_folder, exist_ok=True)
        output_path = os.path.join(final_counts_folder, 'shock-mleed-data.csv')
        combined_mleed.to_csv(output_path, index=False)
        print(f"Combined MLEED shock data saved to: {output_path}")
    else:
        print("No MLEED shock CSV files found to combine.")

print("Combining shock detection files...")
combine_civic_shock_files()
combine_rai_shock_files()
combine_mleed_shock_files()   # <-- NEW
print("File combination completed.")

# =========================
# Build full datasets & merge shocks
# =========================

def _normalize_date_col(df):
    """Force date to month-start; leave untouched if missing."""
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['date'] = df['date'].dt.to_period('M').dt.start_time
    return df

def _rename_shock_cols(shock_df, on_cols):
    """Rename all non-key columns in shock_df to append 'Shock' suffix."""
    rename_map = {}
    for c in shock_df.columns:
        if c not in on_cols:
            rename_map[c] = f"{c}Shock"
    return shock_df.rename(columns=rename_map)

def _left_merge_shocks(full_df, shock_df, on_cols):
    """Left-join and fill missing shock flags with 0; keep ints."""
    if shock_df is None or shock_df.empty:
        return full_df
    merged = full_df.merge(shock_df, on=on_cols, how='left')
    # Fill shocks with 0 and cast to int where possible
    shock_cols = [c for c in merged.columns if c.endswith('Shock')]
    if shock_cols:
        merged[shock_cols] = merged[shock_cols].fillna(0)
        for c in shock_cols:
            # coerce numeric then int (safe)
            merged[c] = pd.to_numeric(merged[c], errors='coerce').fillna(0).astype(int)
    return merged

def build_full_civic_with_shocks():
    """Stack 1-civic-aggregate into full-civic-data.csv and attach civic shocks."""
    src = civic_data_folder
    dest_dir = os.path.join(SCRIPT_DIR, '../../data/final-counts')
    os.makedirs(dest_dir, exist_ok=True)

    frames = []
    for fname in os.listdir(src):
        if not fname.endswith('.csv'):
            continue
        country = fname[:-4]
        df = pd.read_csv(os.path.join(src, fname))
        df['country'] = country
        _normalize_date_col(df)
        frames.append(df)

    if not frames:
        print("[FULL CIVIC] No source files found; skipping.")
        return

    full = pd.concat(frames, ignore_index=True)

    # Preferred ordering: country, date, then the rest
    cols = ['country', 'date'] + [c for c in full.columns if c not in ['country','date']]
    full = full[cols]

    # Attach shocks (if present)
    shock_path = os.path.join(SCRIPT_DIR, '../../data/final-counts/shock-civic-data.csv')
    if os.path.exists(shock_path):
        shock = pd.read_csv(shock_path)
        _normalize_date_col(shock)
        shock = _rename_shock_cols(shock, on_cols=['country','date'])
        full = _left_merge_shocks(full, shock, on_cols=['country','date'])
    else:
        print("[FULL CIVIC] shock-civic-data.csv not found; writing counts only.")

    out = os.path.join(dest_dir, 'full-civic-data.csv')
    full.to_csv(out, index=False)
    print(f"[FULL CIVIC] wrote: {out}")

def build_full_rai_with_shocks():
    """Stack 1-rai-aggregate into full-rai-data.csv and attach RAI shocks."""
    src = rai_data_folder
    dest_dir = os.path.join(SCRIPT_DIR, '../../data/final-counts')
    os.makedirs(dest_dir, exist_ok=True)

    frames = []
    for fname in os.listdir(src):
        if not fname.endswith('.csv') or '_' not in fname:
            continue
        country, influencer = fname[:-4].split('_', 1)
        df = pd.read_csv(os.path.join(src, fname))
        df['country'] = country
        df['influencer'] = influencer
        _normalize_date_col(df)
        frames.append(df)

    if not frames:
        print("[FULL RAI] No source files found; skipping.")
        return

    full = pd.concat(frames, ignore_index=True)

    cols = ['country', 'influencer', 'date'] + [c for c in full.columns if c not in ['country','influencer','date']]
    full = full[cols]

    # Attach shocks (if present)
    shock_path = os.path.join(SCRIPT_DIR, '../../data/final-counts/shock-rai-data.csv')
    if os.path.exists(shock_path):
        shock = pd.read_csv(shock_path)
        _normalize_date_col(shock)
        shock = _rename_shock_cols(shock, on_cols=['country','influencer','date'])
        full = _left_merge_shocks(full, shock, on_cols=['country','influencer','date'])
    else:
        print("[FULL RAI] shock-rai-data.csv not found; writing counts only.")

    out = os.path.join(dest_dir, 'full-rai-data.csv')
    full.to_csv(out, index=False)
    print(f"[FULL RAI] wrote: {out}")

def build_full_mleed_with_shocks():
    """Stack 1-mleed-aggregate into full-mleed-data.csv and attach MLEED shocks."""
    src = mleed_data_folder
    dest_dir = os.path.join(SCRIPT_DIR, '../../data/final-counts')
    os.makedirs(dest_dir, exist_ok=True)

    frames = []
    for fname in os.listdir(src):
        if not fname.endswith('.csv') or '_' in fname:
            # Expect {Country}.csv in 1-mleed-aggregate
            continue
        country = fname[:-4]
        df = pd.read_csv(os.path.join(src, fname))
        df['country'] = country
        _normalize_date_col(df)
        frames.append(df)

    if not frames:
        print("[FULL MLEED] No source files found; skipping.")
        return

    full = pd.concat(frames, ignore_index=True)
    cols = ['country', 'date'] + [c for c in full.columns if c not in ['country','date']]
    full = full[cols]

    # Attach shocks (if present)
    shock_path = os.path.join(SCRIPT_DIR, '../../data/final-counts/shock-mleed-data.csv')
    if os.path.exists(shock_path):
        shock = pd.read_csv(shock_path)
        _normalize_date_col(shock)
        shock = _rename_shock_cols(shock, on_cols=['country','date'])
        full = _left_merge_shocks(full, shock, on_cols=['country','date'])
    else:
        print("[FULL MLEED] shock-mleed-data.csv not found; writing counts only.")

    out = os.path.join(dest_dir, 'full-mleed-data.csv')
    full.to_csv(out, index=False)
    print(f"[FULL MLEED] wrote: {out}")

# Call builders after shock files exist
print("Building full datasets with shock flags merged...")
build_full_civic_with_shocks()
build_full_rai_with_shocks()
build_full_mleed_with_shocks()
print("Done building full datasets.")
