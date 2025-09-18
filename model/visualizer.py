import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def generate_figures(metrics_path, embeddings_path, token_path, output_dir="figures"):
    os.makedirs(output_dir, exist_ok=True)

    # === 1. Métricas de entrenamiento ===
    metrics = pd.read_csv(metrics_path)

    plt.figure(figsize=(8, 4))
    sns.barplot(x="Accuracy", y="Fold", data=metrics, palette="Blues_d")
    plt.title("Accuracy comparison across model architectures")
    plt.xlabel("Accuracy (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison_accuracy.png"))
    plt.close()

    plt.figure(figsize=(8, 4))
    sns.barplot(x="F1 Score", y="Fold", data=metrics, palette="Greens_d")
    plt.title("F1 Score comparison across model architectures")
    plt.xlabel("F1 Score (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison_f1.png"))
    plt.close()

    # === 2. Matriz de confusión global ===
    if "Fold" in metrics.columns and "TP" in metrics.columns:
        all_row = metrics[metrics["Fold"] == "All"]
        if not all_row.empty:
            tn, fp, fn, tp = all_row.iloc[0][["TN", "FP", "FN", "TP"]]
            cm = np.array([[tn, fp], [fn, tp]])
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Control", "Sepsis"])
            fig, ax = plt.subplots(figsize=(5, 4))
            disp.plot(ax=ax)
            plt.title("Confusion matrix for ClinicalBERT-based classification")
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "confusion_matrix_global.png"))
            plt.close()

    # === 3. Distribución de tokens ===
    if os.path.exists(token_path):
        df_tokens = pd.read_csv(token_path)
        token_counts = {}
        for row in df_tokens["prompt"]:
            tokens = row.split("Token sequence: ")[-1].split(".")[0].split()
            for t in tokens:
                token_counts[t] = token_counts.get(t, 0) + 1
        top_tokens = dict(sorted(token_counts.items(), key=lambda x: -x[1])[:40])

        plt.figure(figsize=(12, 4))
        plt.bar(top_tokens.keys(), top_tokens.values(), color='purple')
        plt.xticks(rotation=90)
        plt.title("Frequency distribution of PPG tokens (KMeans cluster output)")
        plt.xlabel("Token ID")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "token_distribution.png"))
        plt.close()

    # === 4. PCA de embeddings ===
    if os.path.exists(embeddings_path):
        df = pd.read_csv(embeddings_path)
        feature_cols = [
            "mean", "std", "min", "max", "skew", "kurtosis", "hr", "hrv",
            "tachycardia", "bradycardia", "hypo_hrv",
            "pulse_range", "pulse_slope", "slope_excess", "range_depressed"
        ]
        df = df.dropna(subset=feature_cols)
        X = df[feature_cols].values
        labels = df["target"].map({0: "Control", 1: "Sepsis"}).values

        X_scaled = StandardScaler().fit_transform(X)
        X_pca = PCA(n_components=2).fit_transform(X_scaled)
        df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
        df_pca["Label"] = labels

        plt.figure(figsize=(6, 5))
        sns.scatterplot(data=df_pca, x="PC1", y="PC2", hue="Label", palette="Set1")
        plt.title("PCA projection of PPG segments with sepsis/control labeling")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "pca_projection_ppg_segments.png"))
        plt.close()

    print(f"✅ All figures saved in '{output_dir}'")

if __name__ == "__main__":
    generate_figures(
        metrics_path="bench/bert_metrics_summary.csv",
        embeddings_path="token_outputs/ppg_segment_embeddings.csv",
        token_path="token_outputs/ppg_token_prompts.csv"
    )
