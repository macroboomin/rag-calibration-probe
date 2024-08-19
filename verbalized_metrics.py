import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

def compute_metrics(df):
    # Expected Calibration Error (ECE)
    def compute_ece(df, n_bins=10):
        bins = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bins[:-1]
        bin_uppers = bins[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = df[(df['confidence_normalized'] > bin_lower) & (df['confidence_normalized'] <= bin_upper)]
            prop_in_bin = len(in_bin) / len(df)
            if len(in_bin) > 0:
                avg_confidence_in_bin = in_bin['confidence_normalized'].mean()
                avg_accuracy_in_bin = in_bin['correct'].mean()
                ece += np.abs(avg_confidence_in_bin - avg_accuracy_in_bin) * prop_in_bin
        
        return ece

    # AUROC
    def compute_auroc(df):
        return roc_auc_score(df['correct'], df['confidence'])

    # AUPRC-Positive (PR-P)
    def compute_pr_p(df):
        precision, recall, _ = precision_recall_curve(df['correct'], df['confidence'])
        return auc(recall, precision)
    
    # AUPRC-Negative (PR-N)
    def compute_pr_n(df):
        precision, recall, _ = precision_recall_curve(1 - df['correct'], df['confidence'])
        return auc(recall, precision)

    # Accuracy
    def compute_accuracy(df):
        return df['correct'].mean()

    # Normalize confidence for ECE calculation
    df['confidence_normalized'] = df['confidence'] / 100

    ece = compute_ece(df) * 100
    auroc = compute_auroc(df) * 100
    pr_p = compute_pr_p(df) * 100
    pr_n = compute_pr_n(df) * 100
    accuracy = compute_accuracy(df) * 100

    return round(ece, 1), round(auroc, 1), round(pr_p, 1), round(pr_n, 1), round(accuracy, 1)

# Load the CSV files
col_math = pd.read_csv('./verbalized_results/Col_Math_verbalized.csv')
biz_ethics = pd.read_csv('./verbalized_results/Biz_Ethics_verbalized.csv')
prf_law = pd.read_csv('./verbalized_results/Prf_Law_verbalized.csv')
com_secu = pd.read_csv('./verbalized_results/Com_Secu_verbalized.csv')
anatomy = pd.read_csv('./verbalized_results/Anatomy_verbalized.csv')
astronomy = pd.read_csv('./verbalized_results/Astronomy_verbalized.csv')
marketing = pd.read_csv('./verbalized_results/Marketing_verbalized.csv')
world_rel = pd.read_csv('./verbalized_results/World_Rel_verbalized.csv')
gsm8k = pd.read_csv('./verbalized_results/GSM8K_verbalized.csv')

# Compute metrics for each dataset
metrics_col_math = compute_metrics(col_math)
metrics_biz_ethics = compute_metrics(biz_ethics)
metrics_prf_law = compute_metrics(prf_law)
metrics_com_secu = compute_metrics(com_secu)
metrics_anatomy = compute_metrics(anatomy)
metrics_astronomy = compute_metrics(astronomy)
metrics_marketing = compute_metrics(marketing)
metrics_world_rel = compute_metrics(world_rel)
metrics_gsm8k = compute_metrics(gsm8k)

# Calculate mean metrics across all datasets
all_metrics = [
    metrics_col_math,
    metrics_biz_ethics,
    metrics_prf_law,
    metrics_com_secu,
    metrics_anatomy,
    metrics_astronomy,
    metrics_marketing,
    metrics_world_rel,
    metrics_gsm8k
]

avg_ece = round(np.mean([m[0] for m in all_metrics]), 1)
avg_auroc = round(np.mean([m[1] for m in all_metrics]), 1)
avg_pr_p = round(np.mean([m[2] for m in all_metrics]), 1)
avg_pr_n = round(np.mean([m[3] for m in all_metrics]), 1)
avg_accuracy = round(np.mean([m[4] for m in all_metrics]), 1)

results = pd.DataFrame({
    'Metric': ['ECE', 'AUROC', 'PR-P', 'PR-N', 'Accuracy'],
    'College Mathematics': metrics_col_math,
    'Business Ethics': metrics_biz_ethics,
    'Professional Law': metrics_prf_law,
    'Computer Security': metrics_com_secu,
    'Anatomy': metrics_anatomy,
    'Astronomy': metrics_astronomy,
    'Marketing': metrics_marketing,
    'World Religion': metrics_world_rel,
    'GSM8K': metrics_gsm8k,
    'Average': [avg_ece, avg_auroc, avg_pr_p, avg_pr_n, avg_accuracy]
})

results.to_csv('./verbalized_results/verbalized_metrics.csv', index=False)

# Output metrics
print("College Mathematics Metrics: ECE, AUROC, PR-P, PR-N, Accuracy")
print(metrics_col_math)
print("Business Ethics Metrics: ECE, AUROC, PR-P, PR-N, Accuracy")
print(metrics_biz_ethics)
print("Professional Law Metrics: ECE, AUROC, PR-P, PR-N, Accuracy")
print(metrics_prf_law)
print("Computer Security Metrics: ECE, AUROC, PR-P, PR-N, Accuracy")
print(metrics_com_secu)
print("Anatomy Metrics: ECE, AUROC, PR-P, PR-N, Accuracy")
print(metrics_anatomy)
print("Astronomy Metrics: ECE, AUROC, PR-P, PR-N, Accuracy")
print(metrics_astronomy)
print("Marketing Metrics: ECE, AUROC, PR-P, PR-N, Accuracy")
print(metrics_marketing)
print("World Religions Metrics: ECE, AUROC, PR-P, PR-N, Accuracy")
print(metrics_world_rel)
print("GSM8K Metrics: ECE, AUROC, PR-P, PR-N, Accuracy")
print(metrics_gsm8k)

print("\nAverage Metrics: ECE, AUROC, PR-P, PR-N, Accuracy")
print(avg_ece, avg_auroc, avg_pr_p, avg_pr_n, avg_accuracy)
