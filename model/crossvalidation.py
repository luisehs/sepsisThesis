import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    precision_score, recall_score
)
from transformers import (
    Trainer, TrainingArguments, AutoTokenizer,
    AutoModelForSequenceClassification, DataCollatorWithPadding
)
from datasets import Dataset
import matplotlib.pyplot as plt
import seaborn as sns

# === Preparar entorno ===
os.makedirs("bench", exist_ok=True)

# === Cargar datos ===
df = pd.read_csv("token_outputs/ppg_token_prompts.csv")
df["label"] = df["completion"].map({"No": 0, "Yes": 1})

# === Balancear casos y controles ===
df_case = df[df["label"] == 1]
df_control = df[df["label"] == 0]
n = min(len(df_case), len(df_control))
df_case = df_case.sample(n, random_state=42)
df_control = df_control.sample(n, random_state=42)
df_balanced = pd.concat([df_case, df_control]).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Dataset balanceado: {len(df_case)} casos y {len(df_control)} controles (total {len(df_balanced)})")

# === Tokenizador ===
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

def tokenize_bert(ex):
    return tokenizer(ex["prompt"], truncation=True, padding="max_length", max_length=512)

# === Validación cruzada tipo 75/25 ===
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.25, random_state=42)

accs, f1s = [], []
all_true, all_pred = [], []
fold_metrics = []

for fold, (train_idx, test_idx) in enumerate(sss.split(df_balanced["prompt"], df_balanced["label"]), 1):
    print(f"\n=== Fold {fold} ===")

    df_train = df_balanced.iloc[train_idx].reset_index(drop=True)
    df_test = df_balanced.iloc[test_idx].reset_index(drop=True)

    # Conteo por clase
    train_counts = df_train["label"].value_counts().to_dict()
    test_counts = df_test["label"].value_counts().to_dict()
    train_control = train_counts.get(0, 0)
    train_case = train_counts.get(1, 0)
    test_control = test_counts.get(0, 0)
    test_case = test_counts.get(1, 0)
    print(f"Train: Control={train_control}, Case={train_case} | Test: Control={test_control}, Case={test_case}")

    # Tokenización
    train_ds = Dataset.from_pandas(df_train[["prompt", "label"]]).map(tokenize_bert)
    test_ds = Dataset.from_pandas(df_test[["prompt", "label"]]).map(tokenize_bert)

    # Modelo
    model = AutoModelForSequenceClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", num_labels=2)

    args = TrainingArguments(
        output_dir=f"./bert_fold_{fold}",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=20,
        evaluation_strategy="epoch",
        save_strategy="no",
        logging_dir=f"./logs_fold_{fold}",
        disable_tqdm=True
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
    )

    trainer.train()
    preds = trainer.predict(test_ds)
    y_true = df_test["label"].values
    y_pred = np.argmax(preds.predictions, axis=1)

    all_true.extend(y_true)
    all_pred.extend(y_pred)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    accs.append(acc)
    f1s.append(f1)

    fold_metrics.append({
        "Fold": fold,
        "Accuracy": acc,
        "F1 Score": f1,
        "Precision": precision,
        "Recall": recall,
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "Train_Control": train_control, "Train_Case": train_case,
        "Test_Control": test_control, "Test_Case": test_case
    })

    # Matriz de confusión por fold
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - Fold {fold}")
    plt.tight_layout()
    plt.savefig(f"bench/bert_confusion_matrix_fold_{fold}.png")
    plt.close()

# === Matriz de confusión global ===
cm_total = confusion_matrix(all_true, all_pred)
tn, fp, fn, tp = cm_total.ravel()
precision_total = precision_score(all_true, all_pred)
recall_total = recall_score(all_true, all_pred)

plt.figure(figsize=(5, 5))
sns.heatmap(cm_total, annot=True, fmt='d', cmap="Greens", xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - All Folds")
plt.tight_layout()
plt.savefig("bench/bert_confusion_matrix_all_folds.png")
plt.close()

# Agregar métricas globales
fold_metrics.append({
    "Fold": "All",
    "Accuracy": accuracy_score(all_true, all_pred),
    "F1 Score": f1_score(all_true, all_pred),
    "Precision": precision_total,
    "Recall": recall_total,
    "TP": tp, "TN": tn, "FP": fp, "FN": fn,
    "Train_Control": None, "Train_Case": None,
    "Test_Control": None, "Test_Case": None
})

# Guardar métricas
metrics_df = pd.DataFrame(fold_metrics)
metrics_df.to_csv("bench/bert_metrics_summary.csv", index=False)

# Gráfico Accuracy y F1
folds = list(range(1, 6))
plt.figure(figsize=(8, 4))
plt.plot(folds, accs, marker='o', label='Accuracy')
plt.plot(folds, f1s, marker='s', label='F1 Score')
plt.xlabel("Fold")
plt.ylabel("Score")
plt.title("Cross-Validation Performance (Bio_ClinicalBERT) - Balanced 75/25")
plt.ylim(0, 1)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("bench/bert_crossval_scores.png")
plt.show()

# Mostrar resumen
print("\n=== Cross-Validation Summary ===")
print(metrics_df)
