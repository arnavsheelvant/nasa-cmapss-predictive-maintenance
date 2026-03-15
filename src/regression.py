"""
Calculates Time-to-Next-Stage (TTNS) and trains/evaluates regressors
to predict it.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.svm import NuSVR
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
from plotting import plot_ttns_scatter

#This function calculates the Time-to-Next-Stage (TTNS) for each data point in the dataset.
# It uses the derived degradation stage column to determine the time remaining until the next stage is reached.
def calculate_ttns_for_dataset(df_processed, stage_col, n_clusters):
    """
    Calculates the Time-to-Next-Stage (TTNS) for each data point.

    Args:
        df_processed (pd.DataFrame): Dataframe with unit_number, time_in_cycles,
                                     and the derived degradation stage column.
        stage_col (str): Name of the column containing the derived degradation stage.
        n_clusters (int): Total number of degradation stages (0 to N-1).

    Returns:
        pd.DataFrame: The input dataframe with a new 'TTNS' column added.
    """
    print("\nCalculating Time-to-Next-Stage (TTNS)")
    df_with_ttns = df_processed.copy()

    # Ensure stage column is integer type
    df_with_ttns[stage_col] = df_with_ttns[stage_col].astype(int)

    # a) Find the first cycle number for each stage for each engine
    print("Finding first cycle per stage for each engine...")
    first_cycle_per_stage = df_with_ttns.groupby(['unit_number', stage_col])['time_in_cycles'].min().unstack()
    # Rename columns for clarity
    first_cycle_per_stage.columns = [f'first_cycle_stage_{i}' for i in range(n_clusters)]

    # b) Get the maximum cycle for each engine (needed if next stage not reached)
    print("Getting max cycles per engine...")
    engine_max_cycles = df_with_ttns.groupby('unit_number')['time_in_cycles'].max()

    # c) Merge the first cycle info back into the main dataframe
    print("Merging first cycle data back...")
    df_with_ttns = pd.merge(df_with_ttns, first_cycle_per_stage, on='unit_number', how='left')

    # d) Define a function to calculate TTNS for a row
    max_stage = n_clusters - 1
    def _calculate_ttns_row(row):
        current_stage = int(row[stage_col])
        current_cycle = row['time_in_cycles']
        unit_num = row['unit_number']

        if current_stage == max_stage:
            return 0 # Already in the final stage

        next_stage = current_stage + 1
        next_stage_col = f'first_cycle_stage_{next_stage}'

        # Get the cycle number when the next stage first occurs for this engine
        next_stage_first_cycle = row.get(next_stage_col, np.nan)

        if pd.isna(next_stage_first_cycle):
            # Engine never reached the next stage within the observed data
            max_cycle_for_engine = engine_max_cycles.loc[unit_num]
            # TTNS is remaining cycles until the end of life for this engine
            return max(0, max_cycle_for_engine - current_cycle)
        else:
            # Engine reached the next stage
            return max(0, next_stage_first_cycle - current_cycle)

    # Apply the function to calculate TTNS
    print("Applying TTNS calculation function...")
    df_with_ttns['TTNS'] = df_with_ttns.apply(_calculate_ttns_row, axis=1)
    print("TTNS calculation complete.")
    print("Sample TTNS values:")
    print(df_with_ttns[['unit_number', 'time_in_cycles', stage_col, 'TTNS']].head())
    print(df_with_ttns[['unit_number', 'time_in_cycles', stage_col, 'TTNS']].tail())
    return df_with_ttns

# This function trains and evaluates regression models to predict TTNS.
# It uses the XGBoost, HistGradientBoostingRegressor, and NuSVR models.
def train_and_evaluate_regressors(df_with_ttns, X_train_class, X_val_class, y_train_class, y_val_class,
                                 random_state, scaler_numeric_class):
    """
    Defines regression features, trains TTNS prediction models (XGBoost, HistGradBoost, NuSVR),
    and evaluates them.

    Args:
        df_with_ttns (pd.DataFrame): Dataframe including the calculated 'TTNS' column.
        X_train_class (pd.DataFrame): Training features from classification phase.
        X_val_class (pd.DataFrame): Validation features from classification phase.
        y_train_class (pd.Series): Training target (stage) from classification phase.
        y_val_class (pd.Series): Validation target (stage) from classification phase.
        random_state (int): Random state for reproducibility.
        scaler_numeric_class (StandardScaler): Fitted scaler from classification phase
                                               (for scaling features consistently).

    Returns:
        tuple: Contains:
            - regressors (dict): Dictionary of trained regressor models.
            - df_val_results_reg (pd.DataFrame): Validation results dataframe updated with TTNS predictions.
            - best_regressor_name (str): Name of the best regressor based on RMSE.
            - regression_results (dict): Dictionary storing metrics for each regressor.
    """
    print("\nPhase 3: Time-to-Next-Stage (TTNS) Prediction Model")

    #Step 1: Define Target (y) for Regression
    # TTNS was calculated on the full dataset before split. Align with train/val indices.
    y_reg = df_with_ttns['TTNS']
    y_train_reg = y_reg.loc[y_train_class.index]
    y_val_reg = y_reg.loc[y_val_class.index]

    print("\nStep 1: Defined Regression Target (TTNS)")
    print(f"Validation TTNS Mean: {y_val_reg.mean():.2f}") # Basic check

    #Step 2: Define Features (X) for Regression
    # Start with the same features used for classification
    X_train_reg = X_train_class.copy()
    X_val_reg = X_val_class.copy()

    # This helps the regressor know the context (stage) for predicting TTNS
    X_train_reg['current_stage_feature'] = y_train_class
    X_val_reg['current_stage_feature'] = y_val_class
    print("Added 'current_stage_feature' to regressor features.")

    #Step 3: Train Regressors
    print("\nStep 3: Train the regressor models (XGBoost, HistGradientBoost, NuSVR)")
    regressors = {}
    regression_results = {}

    # XGBoost Regressor
    print("\nTraining Regressor 1: XGBoost...")
    start_time = time.time()
    try:
        regressor_XGB = xgb.XGBRegressor( #Parameters for XGBoost were tuned using GridSearchCV
            objective='reg:squarederror',
            n_estimators=1200,
            learning_rate=0.009,
            max_depth=11,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method='gpu_hist',
            predictor='gpu_predictor',
            gpu_id=1,
            random_state=random_state,
            n_jobs=-1
        )
        regressor_XGB.fit(X_train_reg, y_train_reg)
        end_time = time.time()
        regressors['xgboost'] = regressor_XGB
        print(f"XGBoost Regressor training completed in {end_time - start_time:.2f} seconds.")
    except Exception as e:
        print(f"Error training XGBoost Regressor: {e}")

    print("\nScaling features for HistGradientBoostingRegressor and NuSVR...")
    scaler_reg_features = StandardScaler()
    # Fit only on training data
    X_train_reg_scaled = scaler_reg_features.fit_transform(X_train_reg)
    # Transform validation data
    X_val_reg_scaled = scaler_reg_features.transform(X_val_reg)
    print("Features scaled.")

    # HistGradientBoostingRegressor
    print("\nTraining Regressor 2: HistGradientBoostingRegressor...")
    start_time = time.time()
    try:
        hgb_regressor = HistGradientBoostingRegressor(random_state=random_state)
        hgb_regressor.fit(X_train_reg_scaled, y_train_reg)
        end_time = time.time()
        print(f"HistGradientBoostingRegressor training completed in {end_time - start_time:.2f} seconds.")
        regressors['HistGradientBoostingRegressor'] = hgb_regressor
    except Exception as e:
        print(f"Error training HistGradientBoostingRegressor: {e}")


    # NuSVR (Support Vector Regression)
    print("\nTraining Regressor 3: NuSVR...")
    print("Warning: NuSVR training can be slow on large datasets.")
    start_time = time.time()
    try:
        nu_svr = NuSVR(nu=0.5, C=15, gamma=0.01) # Parameters were tuned using GridSearchCV
        nu_svr.fit(X_train_reg_scaled, y_train_reg)
        end_time = time.time()
        print(f"NuSVR training completed in {end_time - start_time:.2f} seconds.")
        regressors['NuSVR'] = nu_svr
    except Exception as e:
        print(f"Error training NuSVR: {e}")


    #Step 4: Evaluate Regressors
    print("\nStep 4: Evaluating Regressors on Validation Set")
    df_val_results_reg = X_val_class.copy()
    df_val_results_reg['true_stage'] = y_val_class # Add true stage if needed
    df_val_results_reg['true_ttns'] = y_val_reg   # Add true TTNS

    best_regressor_name = None
    best_rmse = float('inf')

    for name, regressor in regressors.items():
        print(f"\nEvaluation for Regressor: {name.upper()}")

        # Decide which feature set to use for prediction based on the model
        if name in ['HistGradientBoostingRegressor', 'NuSVR']:
            X_pred_features = X_val_reg_scaled
        else: # Assumes XGBoost used unscaled augmented features
             X_pred_features = X_val_reg

        # Predict TTNS
        y_pred_reg = regressor.predict(X_pred_features)
        # Ensure predictions are non-negative
        y_pred_reg = np.maximum(y_pred_reg, 0)

        # Store prediction in the results dataframe
        df_val_results_reg[f'predicted_ttns_{name}'] = y_pred_reg

        # Calculate metrics (only on non-NaN true values if any exist)
        valid_indices = ~np.isnan(y_val_reg)
        if valid_indices.sum() > 0:
             y_val_reg_valid = y_val_reg[valid_indices]
             y_pred_reg_valid = y_pred_reg[valid_indices]

             mse = mean_squared_error(y_val_reg_valid, y_pred_reg_valid)
             rmse = np.sqrt(mse)
             mae = mean_absolute_error(y_val_reg_valid, y_pred_reg_valid)
             r2 = r2_score(y_val_reg_valid, y_pred_reg_valid)

             regression_results[name] = {
                 'rmse': rmse,
                 'mae': mae,
                 'r2': r2,
                 'y_pred': y_pred_reg # Storing full array might use lots of memory
             }

             print("\nRegression Report (Validation Set):")
             print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
             print(f"Mean Absolute Error (MAE):    {mae:.4f}")
             print(f"R-squared (R²) Score:         {r2:.4f}")

             # Update best regressor based on RMSE
             if rmse < best_rmse:
                 best_rmse = rmse
                 best_regressor_name = name

             # Plot True vs Predicted TTNS
             plot_ttns_scatter(y_val_reg_valid, y_pred_reg_valid, name)
        else:
             print("Warning: No valid true TTNS values found for metric calculation.")
             regression_results[name] = {'rmse': np.nan, 'mae': np.nan, 'r2': np.nan}


    if best_regressor_name:
        print(f"\nBest performing regressor based on lowest RMSE: {best_regressor_name.upper()} (RMSE: {best_rmse:.4f})")
        # Add the best prediction as the main 'predicted_ttns' column
        df_val_results_reg['predicted_ttns'] = df_val_results_reg[f'predicted_ttns_{best_regressor_name}']
    else:
        print("\nWarning: Could not determine the best regressor.")
        # Fallback: use the first regressor's prediction or set to NaN
        if regressors:
            first_reg_name = list(regressors.keys())[0]
            if f'predicted_ttns_{first_reg_name}' in df_val_results_reg:
                 df_val_results_reg['predicted_ttns'] = df_val_results_reg[f'predicted_ttns_{first_reg_name}']
                 best_regressor_name = first_reg_name # Assume first is best if none clearly better
            else:
                 df_val_results_reg['predicted_ttns'] = np.nan
        else:
            df_val_results_reg['predicted_ttns'] = np.nan

    print("\nPhase 3 Complete")
    print("TTNS calculated and regression models trained/evaluated.")
    return regressors, df_val_results_reg, best_regressor_name, regression_results