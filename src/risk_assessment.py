"""
Calculates the raw and normalized risk scores based on classification
probabilities and TTNS predictions, and applies a threshold for alerts.
"""
import pandas as pd
import numpy as np

# This function calculates the risk score based on the probability of failure
def calculate_and_evaluate_risk(df_val_results_reg, best_classifier_name, n_clusters, risk_threshold):
    """
    Calculates risk scores, normalizes them, generates alerts, and analyzes results.

    Args:
        df_val_results_reg (pd.DataFrame): Dataframe containing validation results
                                           (true stage, classifier probabilities, predicted TTNS).
        best_classifier_name (str): Name of the selected best classifier.
        n_clusters (int): Number of degradation stages.
        risk_threshold (float): Threshold for triggering a maintenance alert (0 to 1).

    Returns:
        pd.DataFrame: The input dataframe updated with risk scores and alerts.
    """
    print("\nPhase 4: Risk Score Calculation and Evaluation")
    df_risk = df_val_results_reg.copy()

    #Step 1: Calculate Raw Risk Score
    print("\nStep 1: Calculate Raw Risk Score")
    failure_stage = n_clusters - 1 # last stage is the failure stage (e.g., 4 for FD001/FD003)
    # Construct the column name for the probability of the failure stage from the best classifier
    failure_stage_prob_col = f'prob_stage_{failure_stage}_{best_classifier_name}'
    time_left_col = 'predicted_ttns' # Using the prediction from the best regressor
    epsilon = 1e-6 # Small constant to prevent division by zero

    # Check if necessary columns exist
    if failure_stage_prob_col not in df_risk.columns:
        print(f"Error: Probability column '{failure_stage_prob_col}' not found in results. Cannot calculate risk score.")
        # Assign default/NaN values or raise error
        df_risk['raw_risk_score'] = np.nan
        df_risk['normalized_risk_score'] = np.nan
        df_risk['maintenance_alert'] = False
        return df_risk # Return early

    if time_left_col not in df_risk.columns:
        print(f"Error: Predicted TTNS column '{time_left_col}' not found in results. Cannot calculate risk score.")
        df_risk['raw_risk_score'] = np.nan
        df_risk['normalized_risk_score'] = np.nan
        df_risk['maintenance_alert'] = False
        return df_risk # Return early

    print(f"Calculating Raw Risk Score using: Probability('{failure_stage_prob_col}') / (TTNS('{time_left_col}') + epsilon)")

    # Calculate raw score: Higher probability of failure / Shorter time left = Higher risk
    # Ensure inputs are numeric
    prob_failure = pd.to_numeric(df_risk[failure_stage_prob_col], errors='coerce').fillna(0)
    pred_ttns = pd.to_numeric(df_risk[time_left_col], errors='coerce').fillna(epsilon) # Handle potential NaNs in TTNS

    df_risk['raw_risk_score'] = prob_failure / (pred_ttns + epsilon)

    # Handle potential infinite values resulting from division by ~zero TTNS
    if np.isinf(df_risk['raw_risk_score']).any():
        print("Handling infinite values in raw_risk_score...")
        max_finite_risk = df_risk.loc[np.isfinite(df_risk['raw_risk_score']), 'raw_risk_score'].max()
        if pd.isna(max_finite_risk): max_finite_risk = 1.0
        replacement_value = max_finite_risk * 1.1 if max_finite_risk > 0 else 1.0
        df_risk['raw_risk_score'] = df_risk['raw_risk_score'].replace([np.inf, -np.inf], replacement_value)
        print(f"Infinite values replaced or handled. Max finite risk approx: {max_finite_risk:.4f}")

    # Ensure score is non-negative
    df_risk['raw_risk_score'] = df_risk['raw_risk_score'].clip(lower=0)


    #Step 2: Normalize Risk Score (Min-Max Scaling)
    print("\nStep 2: Normalize Risk Score using Min-Max Scaling...")
    min_risk = df_risk['raw_risk_score'].min()
    max_risk = df_risk['raw_risk_score'].max()

    print(f"Raw Risk Score Range: Min={min_risk:.4f}, Max={max_risk:.4f}")

    # Avoid division by zero if all scores are identical
    if max_risk > min_risk:
        df_risk['normalized_risk_score'] = (df_risk['raw_risk_score'] - min_risk) / (max_risk - min_risk)
    else:
        # If min equals max, set normalized score based on whether the score is non-zero
        df_risk['normalized_risk_score'] = 0.0 if min_risk == 0 else 1.0
        print(f"Warning: Raw risk scores have zero range (all are {min_risk:.4f}). Normalized score set accordingly.")

    # Ensure normalized score is within [0, 1] after potential float inaccuracies
    df_risk['normalized_risk_score'] = df_risk['normalized_risk_score'].clip(0, 1)

    print(f"Normalized Risk Score Range: Min={df_risk['normalized_risk_score'].min():.4f}, Max={df_risk['normalized_risk_score'].max():.4f}")

    #Step 3: Decision Making (Maintenance Alert)
    print("\nStep 3: Apply Maintenance Alert Threshold")
    print(f"Using Risk Threshold: {risk_threshold}")

    df_risk['maintenance_alert'] = df_risk['normalized_risk_score'] > risk_threshold

    # Analyze Alerts
    alert_counts = df_risk['maintenance_alert'].value_counts()
    print("\nMaintenance Alert Summary (Validation Set):")
    print(alert_counts)

    # Check if 'true_stage' column exists before detailed analysis
    if 'true_stage' in df_risk.columns and not df_risk['true_stage'].isnull().all():
         if True in alert_counts.index:
             # Analyze alerts by true stage
             print("\nAlerts triggered by True Degradation Stage:")
             alerts_by_stage = df_risk[df_risk['maintenance_alert']]['true_stage'].value_counts().sort_index()
             print(alerts_by_stage)
         else:
             print("No alerts triggered with the current threshold.")

         # Analyze Stage 4 (Failure) Alerts
         stage4_total = df_risk[df_risk['true_stage'] == failure_stage].shape[0]

         if stage4_total > 0:
             stage4_alerts_triggered = df_risk[
                 (df_risk['true_stage'] == failure_stage) &
                 (df_risk['maintenance_alert'] == True)
             ].shape[0]

             print(f"\nAnalysis of Alerts for True Stage {failure_stage} (Failure):")
             print(f"Total data points in true_stage {failure_stage}: {stage4_total}")
             print(f"Alerts triggered for true_stage {failure_stage} points: {stage4_alerts_triggered}")

             stage4_recall = (stage4_alerts_triggered / stage4_total) * 100
             print(f"Percentage of true_stage {failure_stage} points that triggered an alert (Recall for Stage {failure_stage}): {stage4_recall:.2f}%")
         else:
             print(f"\nNo data points found for True Stage {failure_stage} in the validation set.")

    else:
         print("\nWarning: 'true_stage' column not available or empty. Skipping detailed alert analysis by stage.")


    print("\nPhase 4 Complete")
    print("Risk score calculated, normalized, and alerts generated.")
    return df_risk