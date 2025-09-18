import pandas as pd
import numpy as np
import os
import scipy.io
from collections import defaultdict
from scipy.signal import find_peaks
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# === CONFIGURACIÓN ===
WINDOW_SIZE = 250
STEP_SIZE = 125
N_PCA_COMPONENTS = 0.95
N_CLUSTERS = 128
OUTPUT_DIR = "token_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === FUNCIONES ===

def segment_signal(signal, window_size=250, step_size=125):
    return np.array([
        signal[i:i + window_size]
        for i in range(0, len(signal) - window_size + 1, step_size)
    ])

def extract_features_from_segments(segments, sampling_rate=125):
    features = []
    for seg in segments:
        peaks, _ = find_peaks(seg, distance=sampling_rate//2)
        ibi = np.diff(peaks) / sampling_rate if len(peaks) > 1 else [0]
        hr = 60 / np.mean(ibi) if np.mean(ibi) > 0 else 0
        hrv = np.std(ibi) if len(ibi) > 1 else 0

        # SIRS/SOFA-inspired
        tachycardia = 1 if hr > 90 else 0
        bradycardia = 1 if hr < 60 else 0
        hypo_hrv = 1 if hrv < 0.05 else 0
        pulse_range = np.max(seg) - np.min(seg)
        pulse_slope = np.mean(np.abs(np.diff(seg)))
        slope_excess = 1 if pulse_slope > 1.0 else 0
        range_depressed = 1 if pulse_range < 0.2 else 0

        feats = [
            np.mean(seg), np.std(seg), np.min(seg), np.max(seg),
            skew(seg), kurtosis(seg), hr, hrv,
            tachycardia, bradycardia, hypo_hrv,
            pulse_range, pulse_slope, slope_excess, range_depressed
        ]
        features.append(feats)
    return np.array(features)

def process_all_signals(signals, targets, subject_ids):
    records = []
    for signal, target, sid in zip(signals, targets, subject_ids):
        segments = segment_signal(signal)
        embeddings = extract_features_from_segments(segments)
        for emb in embeddings:
            record = dict(zip([
                "mean", "std", "min", "max", "skew", "kurtosis", "hr", "hrv",
                "tachycardia", "bradycardia", "hypo_hrv",
                "pulse_range", "pulse_slope", "slope_excess", "range_depressed"
            ], emb))
            record.update({"target": int(target), "subject_id": int(sid)})
            records.append(record)
    return pd.DataFrame(records)

def create_token_sequences(df_embeddings, n_components=0.95, n_clusters=128):
    feature_cols = [
        "mean", "std", "min", "max", "skew", "kurtosis", "hr", "hrv",
        "tachycardia", "bradycardia", "hypo_hrv",
        "pulse_range", "pulse_slope", "slope_excess", "range_depressed"
    ]
    df_features = df_embeddings[feature_cols + ["subject_id", "target"]].dropna()
    X = df_features[feature_cols].values
    subject_ids = df_features["subject_id"].values
    targets = df_features["target"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(6, 4))
    plt.plot(np.cumsum(PCA().fit(X_scaled).explained_variance_ratio_) * 100, marker='o')
    plt.title("Cumulative Explained Variance by PCA Components")
    plt.xlabel("Number of Components")
    plt.ylabel("Explained Variance (%)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "pca_variance.png"))
    plt.close()

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    tokens = kmeans.fit_predict(X_pca)

    sns.countplot(x=tokens)
    plt.title("Distribution of Tokens (KMeans Clusters)")
    plt.xlabel("Token ID")
    plt.ylabel("Frequency")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "token_distribution.png"))
    plt.close()

    subject_token_map = defaultdict(list)
    subject_target_map = {}
    for token, sid, label in zip(tokens, subject_ids, targets):
        subject_token_map[sid].append(f"tok_{token}")
        subject_target_map[sid] = label

    records = []
    for sid, token_list in subject_token_map.items():
        sequence = " ".join(token_list)
        label = subject_target_map[sid]
        prompt = f"Token sequence: {sequence}. Is this patient septic?"
        completion = "Yes" if label == 1 else "No"
        records.append({"subject_id": sid, "prompt": prompt, "completion": completion})

    return pd.DataFrame(records)

# === EJECUCIÓN PRINCIPAL ===
if __name__ == "__main__":
    print("Cargando archivo .mat...")
    mat = scipy.io.loadmat("/root/thesis_lh/data/datamat/22_1min.mat")
    signals = mat['signal']
    targets = mat['target'].flatten()
    if isinstance(targets[0], str) or isinstance(targets[0], np.str_):
        targets_lower = np.char.lower(targets.astype(str))
        assert np.all(np.isin(targets_lower, ['control', 'case   '])), f"Valores no válidos en target: {set(targets_lower)}"
        targets = np.where(targets_lower == 'control', 0, 1)

    elif isinstance(targets[0], (int, np.integer)):
        pass
    subject_ids = mat['subject'].flatten()

    print("Extrayendo características...")
    df_embeddings = process_all_signals(signals, targets, subject_ids)
    df_embeddings.to_csv(os.path.join(OUTPUT_DIR, "ppg_segment_embeddings.csv"), index=False)

    print("Generando secuencias tokenizadas...")
    df_tokens = create_token_sequences(df_embeddings, n_components=N_PCA_COMPONENTS, n_clusters=N_CLUSTERS)
    df_tokens.to_csv(os.path.join(OUTPUT_DIR, "ppg_token_prompts.csv"), index=False)

    print("Listo. Prompt tokens guardados en:", os.path.join(OUTPUT_DIR, "ppg_token_prompts.csv"))
