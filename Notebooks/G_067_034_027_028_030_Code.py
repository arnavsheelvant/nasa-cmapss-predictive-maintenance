# %% [markdown]
#  # Multi-Stage Failure Labeling and Risk Assessment Pipeline
# 
# 
# 
#  This notebook orchestrates the process:
# 
#  1. **Load & Prepare Data:** Reads CMAPSS data, handles multiple files, creates unique IDs.
# 
#  2. **Clustering & Staging:** Scales sensors, applies KMeans/GMM, interprets clusters as degradation stages based on time.
# 
#  3. **Classification:** Trains models (XGBoost, LogReg, SVC) to predict the derived degradation stage.
# 
#  4. **Regression:** Calculates Time-to-Next-Stage (TTNS) and trains models (XGBoost, HistGB, NuSVR) to predict it.
# 
#  5. **Risk Assessment:** Combines stage prediction probability and TTNS prediction into a normalized risk score and generates maintenance alerts.
# 
#  6. **Visualization:** Shows results at each step (distributions, PCA, trends, confusion matrices, TTNS plots, risk plots).

# %% [markdown]
#  ## Setup and Configuration
# 
#  Change which Dataset or Datasets you want to use in the config.py file

# %%
import pandas as pd
import numpy as np
import warnings
import time

# Import custom modules
from config import * # Load all constants from config.py
from data_loader import load_and_preprocess_raw_data
from plotting import (plot_cluster_distribution, plot_pca_clusters,
                    plot_sensor_trends_by_stage,
                    plot_risk_trends_sample_engines, plot_and_save_engine_risk_trends)
from clustering import identify_and_scale_sensors, run_clustering_phase
from classification import train_and_evaluate_classifiers # Rolling features handled internally
from regression import calculate_ttns_for_dataset, train_and_evaluate_regressors
from risk_assessment import calculate_and_evaluate_risk


# Configure warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

# Display Configuration
print(f"Using Dataset Path: {DATASET_PATH}")
print(f"Using Dataset IDs: {DATASET_IDS}")
print(f"Combined Dataset Name: {COMBINED_DATASET_NAME}")
print(f"Number of Degradation Stages to derive: {N_CLUSTERS}")
print(f"Random State: {RANDOM_STATE}")
print(f"Risk Threshold for Alerts: {RISK_THRESHOLD}")
print(f"Rolling Feature Window Size: {ROLLING_WINDOW_SIZE}")


# %% [markdown]
#  ## Phase 0: Data Loading and Initial Preparation

# %%
start_time_total = time.time()

# Load data using the dedicated function
df_train_raw = load_and_preprocess_raw_data(DATASET_PATH, DATASET_IDS)

if df_train_raw is None:
    print("Stopping execution due to data loading errors.")
else:
    print("\nRaw data loaded successfully.")
    # Define column sets based on the final loaded dataframe
    all_cols = df_train_raw.columns
    sensor_cols = [col for col in all_cols if col.startswith('sensor_')]
    op_setting_cols = [col for col in all_cols if col.startswith('operational_setting_')]
    print(f"Identified {len(sensor_cols)} sensor columns.")
    print(f"Identified {len(op_setting_cols)} operational setting columns.")

# %% [markdown]
#  ## Phase 1: Clustering for Multi-Stage Failure Labeling

# %%
if df_train_raw is not None:
    # Step 1a: Identify low variance sensors and scale active ones
    df_train_processed, scaler_sensors, active_sensor_cols = identify_and_scale_sensors(
        df_train_raw, sensor_cols # Pass the original list of sensor columns
    )
    print(f"\nUsing {len(active_sensor_cols)} active sensor columns for clustering.")

    # Step 1b: Run KMeans and GMM clustering, interpret stages, run PCA
    (df_train_clustered, kmeans_model, gmm_model,
     kmeans_stage_map, gmm_stage_map,
     pca, X_pca) = run_clustering_phase(
        df_train_processed, active_sensor_cols, N_CLUSTERS, RANDOM_STATE
     )

    #Visualize Clustering Results
    print("\nVisualizing Clustering Results")

    # 1. Distribution plots (KMeans derived stage is primary: 'degradation_stage')
    print("Plotting distribution for KMeans derived stages ('degradation_stage')...")
    plot_cluster_distribution(df_train_clustered, 'degradation_stage', N_CLUSTERS)
    print("Plotting distribution for GMM derived stages ('gmm_stage')...")
    plot_cluster_distribution(df_train_clustered, 'gmm_stage', N_CLUSTERS)


    # 2. PCA plots
    print("\nPlotting PCA for KMeans derived stages...")
    plot_pca_clusters(X_pca, df_train_clustered['degradation_stage'], N_CLUSTERS)
    print("Plotting PCA for GMM derived stages...")
    plot_pca_clusters(X_pca, df_train_clustered['gmm_stage'], N_CLUSTERS)


    # 3. Sensor Trends Visualization for sample engines
    print("\nVisualizing sensor trends colored by derived degradation stage (KMeans)...")
    # Select sensors to plot (use defaults, filter by active)
    sensors_to_plot_active = [s for s in DEFAULT_SENSORS_TO_PLOT if s in active_sensor_cols]
    if not sensors_to_plot_active: # Fallback if default sensors were dropped
        sensors_to_plot_active = active_sensor_cols[:min(len(active_sensor_cols), 6)]

    # Select sample engines intelligently across datasets if possible
    unique_engine_ids = df_train_clustered['unit_number'].unique()
    sample_engines_list = []
    if len(DATASET_IDS) > 1:
         for d_id in DATASET_IDS:
             engines_from_dset = [eng for eng in unique_engine_ids if eng.startswith(d_id + '_')]
             sample_engines_list.extend(np.random.choice(engines_from_dset, min(len(engines_from_dset), 2), replace=False)) # Pick 2 random from each
         PLOT_SAMPLE_ENGINES = sample_engines_list[:min(len(sample_engines_list), MAX_PLOT_SAMPLE_ENGINES)] # Limit total samples
    else: # Only one dataset loaded
         PLOT_SAMPLE_ENGINES = list(np.random.choice(unique_engine_ids, min(len(unique_engine_ids), MAX_PLOT_SAMPLE_ENGINES), replace=False))

    if not PLOT_SAMPLE_ENGINES: # Handle case where no engines are available
         print("Warning: No sample engines could be selected for trend plotting.")
    elif not sensors_to_plot_active:
        print("Warning: No active sensors available to plot trends.")
    else:
        plot_sensor_trends_by_stage(
            df_train_clustered, sensors_to_plot_active, 'degradation_stage', PLOT_SAMPLE_ENGINES, N_CLUSTERS
        )
        print("\nVisualizing sensor trends colored by derived degradation stage (GMM)...")
        plot_sensor_trends_by_stage(
           df_train_clustered, sensors_to_plot_active, 'gmm_stage', PLOT_SAMPLE_ENGINES, N_CLUSTERS
        )


# %% [markdown]
#  ## Phase 2: Classification Model (Predicting Degradation Stage)
# 
#  Uses the primary derived stage ('degradation_stage' from KMeans) as the target. Includes rolling features.

# %%
if 'df_train_clustered' in locals():
     # Target is the primary degradation stage derived from KMeans
     target_stage_col = 'degradation_stage'

     (classifiers, df_val_pred_staging, X_train_class, X_val_class, y_train_class, y_val_class,
      best_classifier_name, classification_accuracies, scaler_numeric_class) = train_and_evaluate_classifiers(
          df_train_clustered, # Pass the dataframe from clustering
          active_sensor_cols,
          op_setting_cols,
          target_col=target_stage_col,
          n_clusters=N_CLUSTERS,
          random_state=RANDOM_STATE,
          include_rolling_features=True, # Use rolling features
          window_size=ROLLING_WINDOW_SIZE
      )

     # df_val_pred_staging now contains features, true stage, predicted stages, and probabilities
     print("\nValidation Set Predictions (Staging) Head:")
     print(df_val_pred_staging.head())
else:
     print("Skipping Phase 2 because Phase 1 did not complete successfully.")

# %% [markdown]
#  ## Phase 3: Regression Model (Predicting Time-to-Next-Stage - TTNS)

# %%
if 'df_train_clustered' in locals() and 'best_classifier_name' in locals():
    # Step 3a: Calculate TTNS on the *original* clustered dataframe
    # Use the same target stage column as used for classification
    df_with_ttns = calculate_ttns_for_dataset(df_train_clustered, target_stage_col, N_CLUSTERS)

    # Step 3b: Train and evaluate regressors using the splits from classification phase
    # Pass the dataframe with TTNS, the classification features/targets splits, and the numeric scaler
    (regressors, df_val_pred_ttns, best_regressor_name, regression_results) = train_and_evaluate_regressors(
         df_with_ttns, # Contains the TTNS column aligned by index
         X_train_class, X_val_class, # Features from classification split
         y_train_class, y_val_class, # Stage targets from classification split
         RANDOM_STATE,
         scaler_numeric_class # Scaler fitted on numeric features in classification
     )

    # df_val_pred_ttns now contains validation features, true stage, true TTNS, and predicted TTNS for various models + the best one
    print("\nValidation Set Predictions (TTNS) Head:")
    # Merge the staging predictions into the TTNS results dataframe based on index
    cols_to_merge = ['unit_number', 'time_in_cycles', f'predicted_stage_{best_classifier_name}'] + \
                    [f'prob_stage_{i}_{best_classifier_name}' for i in range(N_CLUSTERS) if f'prob_stage_{i}_{best_classifier_name}' in df_val_pred_staging.columns]

    # Check if staging results df exists and has the necessary columns
    if 'df_val_pred_staging' in locals() and all(c in df_val_pred_staging for c in cols_to_merge):
         df_val_final = df_val_pred_ttns.merge(
             df_val_pred_staging[cols_to_merge],
             left_index=True,
             right_index=True,
             how='left'
         )
         print("Merged staging predictions with TTNS results.")
    else:
         print("Warning: Could not merge staging predictions, df_val_final will only contain TTNS results.")
         df_val_final = df_val_pred_ttns # Use only TTNS results if merge fails

    print(df_val_final.head())
else:
     print("Skipping Phase 3 because Phase 1 or 2 did not complete successfully.")

# %% [markdown]
#  ## Phase 4: Risk Score Calculation and Alerting

# %%
if 'df_val_final' in locals() and best_classifier_name is not None and best_regressor_name is not None:
    # Calculate risk using the best classifier's probabilities and best regressor's TTNS
    df_val_final = calculate_and_evaluate_risk(
        df_val_final,
        best_classifier_name,
        N_CLUSTERS,
        RISK_THRESHOLD
    )

    #Visualize Risk Score
    print("\nVisualizing Risk Score Results")
    # 1. Plot risk trends for the same sample engines used in Phase 1
    plot_risk_trends_sample_engines(
        df_val_final, PLOT_SAMPLE_ENGINES, N_CLUSTERS, RISK_THRESHOLD
    )

    # 2. Plot and save risk trends for ALL engines in the validation set
    plot_and_save_engine_risk_trends(
        df_val_final,
        "engine_risk_plots", # Base directory name
        COMBINED_DATASET_NAME,
        N_CLUSTERS,
        RISK_THRESHOLD
    )
else:
     print("Skipping Phase 4 because prior phases did not complete successfully or required data is missing.")


# %% [markdown]
#  ## Pipeline Complete

# %%
end_time_total = time.time()
print(f"\nEntire Pipeline Execution Finished")
print(f"Total execution time: {end_time_total - start_time_total:.2f} seconds")

# Final results are in the df_val_final DataFrame (if all phases ran)
if 'df_val_final' in locals():
    print("\nFinal Validation Results DataFrame head:")
    print(df_val_final.head())
    print("\nFinal Validation Results DataFrame info:")
    df_val_final.info()


