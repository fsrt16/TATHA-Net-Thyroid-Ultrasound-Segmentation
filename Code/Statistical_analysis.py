"""
Statistical Analysis of Machine Learning Model with 15-Fold Cross-Validation
-----------------------------------------------------------------------------

This script:
- Performs 15-fold cross-validation
- Calculates accuracy and F1-score
- Computes mean, standard deviation
- Performs Shapiro-Wilk normality test
- Performs one-sample T-test
- Summarizes results in a dataframe format

Author: Your Name
Date: 2025-06-26
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer, accuracy_score, f1_score
from scipy.stats import shapiro, ttest_1samp

def perform_statistical_analysis(model, X, y):
    """
    Perform 15-fold CV with accuracy and F1 score, 
    compute statistical metrics, normality test, and t-test.
    
    Parameters:
    - model: sklearn-compatible model
    - X: Feature matrix
    - y: Target vector
    
    Returns:
    - results_df: DataFrame with Mean, Std, P-value, T-Test Result
    - raw_scores: Dictionary of all raw scores
    """

    cv = StratifiedKFold(n_splits=15, shuffle=True, random_state=42)

    print("Running 15-Fold Cross-Validation...")
    
    acc_scores = cross_val_score(model, X, y, cv=cv, scoring=make_scorer(accuracy_score))
    f1_scores = cross_val_score(model, X, y, cv=cv, scoring=make_scorer(f1_score, average='weighted'))

    results = {}

    for name, scores in zip(['Accuracy', 'F1 Score'], [acc_scores, f1_scores]):
        mean_val = np.mean(scores)
        std_val = np.std(scores)

        # Shapiro-Wilk test for normality
        shapiro_stat, shapiro_p = shapiro(scores)

        # One-sample T-Test: Are the scores significantly different from chance? (e.g., 0.5)
        t_stat, t_pval = ttest_1samp(scores, 0.5)

        t_result = "Significant" if t_pval < 0.05 else "Not Significant"

        results[name] = {
            'Mean': mean_val,
            'Standard Deviation': std_val,
            'Shapiro P-Value': shapiro_p,
            'T-Value': t_stat,
            'T-Test P-Value': t_pval,
            'T-Test Result': t_result
        }

    results_df = pd.DataFrame(results).T
    return results_df, {'accuracy_scores': acc_scores, 'f1_scores': f1_scores}


# Example usage (To be removed in GitHub version, or kept under `if __name__ == '__main__':`)
if __name__ == '__main__':
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier

    print("Loading dataset...")
    data = load_iris()
    X, y = data.data, data.target

    print("Training RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    results_df, raw_scores = perform_statistical_analysis(model, X, y)
    
    print("\nFinal Summary of 15-Fold Cross-Validation Statistical Analysis:")
    print(results_df.round(4))
