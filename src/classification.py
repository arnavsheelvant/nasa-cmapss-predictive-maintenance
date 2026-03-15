"""
Handles the training and evaluation of classifiers to predict degradation stages.
"""
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import time
from plotting import plot_confusion_matrix_heatmap, plot_feature_importances

# This function adds rolling mean and std features for active sensors.
# It uses a specified window size and fills initial NaNs using backfill and forward fill.
def add_rolling_features(df, active_sensor_cols, window_size):
    """Adds rolling mean and std features for active sensors."""
    print(f"\nAdding rolling window features (window size: {window_size})...")
    df_out = df.copy()
    rolling_features_list = []
    for col in active_sensor_cols:
        if col in df_out.columns:
            mean_col = f'{col}_roll_mean_{window_size}'
            std_col = f'{col}_roll_std_{window_size}'
            df_out[mean_col] = df_out.groupby('unit_number')[col].rolling(window=window_size, min_periods=1).mean().reset_index(level=0, drop=True)
            df_out[std_col] = df_out.groupby('unit_number')[col].rolling(window=window_size, min_periods=1).std().reset_index(level=0, drop=True)
            rolling_features_list.extend([mean_col, std_col])
        else:
            print(f"Warning: Column {col} not found for rolling feature calculation.")

    # Fill initial NaNs from rolling window (using backfill then forward fill within each group)
    if rolling_features_list:
         df_out[rolling_features_list] = df_out.groupby('unit_number')[rolling_features_list].fillna(method='bfill').fillna(method='ffill')
         # Handle any remaining NaNs if an engine has fewer cycles than window size
         df_out[rolling_features_list] = df_out[rolling_features_list].fillna(0) # Fill remaining NaNs with 0
         print(f"Added {len(rolling_features_list)} rolling window features.")
    else:
         print("No rolling features were added.")
    return df_out, rolling_features_list

# This function trains and evaluates classifiers (XGBoost, Logistic Regression, SVC) on the processed data.
# It handles feature selection, scaling, and evaluation metrics.
def train_and_evaluate_classifiers(df_processed, active_sensor_cols, op_setting_cols,
                                   target_col, n_clusters, random_state,
                                   include_rolling_features=True, window_size=50):
    """
    Defines features, splits data, scales non-sensor features, trains classifiers
    (XGBoost, LogReg, SVC), and evaluates them.

    Args:
        df_processed (pd.DataFrame): Dataframe containing processed features and target.
        active_sensor_cols (list): List of scaled active sensor columns.
        op_setting_cols (list): List of operational setting columns.
        target_col (str): Name of the target column (e.g., 'degradation_stage').
        n_clusters (int): Number of target classes (stages).
        random_state (int): Random state for reproducibility.
        include_rolling_features (bool): Whether to calculate and use rolling features.
        window_size (int): Window size if rolling features are used.

    Returns:
        tuple: Contains:
            - classifiers (dict): Dictionary of trained classifier models.
            - df_val_results (pd.DataFrame): Validation data with predictions and probabilities.
            - X_train (pd.DataFrame): Training features.
            - X_val (pd.DataFrame): Validation features.
            - y_train (pd.Series): Training target.
            - y_val (pd.Series): Validation target.
            - best_classifier_name (str): Name of the best performing classifier based on accuracy.
            - classification_accuracies (dict): Accuracies of all classifiers.
            - scaler_numeric (StandardScaler): Fitted scaler for numeric (non-sensor) features.
    """
    print("\nPhase 2: Classification Model (Predicting Degradation Stage)")

    df_class_input = df_processed.copy()
    rolling_features_list = []
    if include_rolling_features:
        df_class_input, rolling_features_list = add_rolling_features(df_class_input, active_sensor_cols, window_size)

    #Step 1: Define Features (X) and Target (y)
    features_to_use = active_sensor_cols + op_setting_cols + rolling_features_list
    # Ensure all feature columns actually exist in the dataframe
    features_to_use = [f for f in features_to_use if f in df_class_input.columns]
    numeric_features_to_scale = op_setting_cols + rolling_features_list
    numeric_features_to_scale = [f for f in numeric_features_to_scale if f in df_class_input.columns]

    target = target_col

    print("\nStep 1: Define Features (X) and Target (y)")
    X = df_class_input[features_to_use]
    y = df_class_input[target]
    print(f"Using target column: '{target}'")
    print(f"Features ({len(features_to_use)}): {features_to_use[:5]}...{features_to_use[-2:]}")
    print(f"Target: {target}")

    #Step 2: Split Data and Scale Non-Sensor Features
    print("\nStep 2: Split into train and test data and scale")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,       # 80% training, 20% validation
        random_state=random_state,
        stratify=y           # Crucial for imbalanced classes
    )
    print(f"Data split into Train ({X_train.shape[0]} samples) and Validation ({X_val.shape[0]} samples)")

    scaler_numeric = StandardScaler()
    if numeric_features_to_scale:
         print(f"Scaling {len(numeric_features_to_scale)} numeric features...")
         # Important: Fit on train, transform both train and val
         X_train[numeric_features_to_scale] = scaler_numeric.fit_transform(X_train[numeric_features_to_scale])
         X_val[numeric_features_to_scale] = scaler_numeric.transform(X_val[numeric_features_to_scale])
         print("Numeric features scaled.")
    else:
         print("No numeric features to scale (OpSettings/Rolling features might be missing or not used).")


    #Step 3: Train Classifiers
    print("\nStep 3: Train the classifier models (XGBoost, Logistic Regression, Support Vector Classifier)")
    classifiers = {}

    # XGBoost
    print("\nTraining Classifier 1: XGBoost...")
    start_time = time.time()
    try:
        xgb_classifier = xgb.XGBClassifier( #Parameters for XGBoost were tuned using GridSearchCV
            objective='multi:softmax',
            num_class=n_clusters,
            n_estimators=800,
            learning_rate=0.1,
            max_depth=13,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            reg_alpha=0.01,
            reg_lambda=0.5,
            use_label_encoder=False,
            eval_metric='mlogloss',
            random_state=random_state,
            n_jobs=-1,
            tree_method='gpu_hist',
            predictor='gpu_predictor',
            gpu_id=1
        )
        xgb_classifier.fit(X_train, y_train)
        end_time = time.time()
        print(f"XGBoost training completed in {end_time - start_time:.2f} seconds.")
        classifiers['xgboost'] = xgb_classifier
    except Exception as e:
        print(f"Error training XGBoost: {e}")


    # Logistic Regression
    print("\nTraining Classifier 2: Logistic Regression...")
    start_time = time.time()
    try:
        logreg_classifier = LogisticRegression(
            random_state=random_state,
            multi_class='ovr',
            solver='liblinear',
            max_iter=1000,
            class_weight='balanced',
            n_jobs=-1
        )
        logreg_classifier.fit(X_train, y_train)
        end_time = time.time()
        print(f"Logistic Regression training completed in {end_time - start_time:.2f} seconds.")
        classifiers['logistic_regression'] = logreg_classifier
    except Exception as e:
        print(f"Error training Logistic Regression: {e}")


    # Support Vector Classifier (SVC)
    print("\nTraining Classifier 3: Support Vector Classifier (SVC)...")
    print("Warning: SVC training can be slow, especially with probability=True.")
    start_time = time.time()
    try:
        svc_classifier = SVC(
            random_state=random_state,
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            class_weight='balanced'
        )
        svc_classifier.fit(X_train, y_train)
        end_time = time.time()
        print(f"SVC training completed in {end_time - start_time:.2f} seconds.")
        classifiers['svc'] = svc_classifier
    except Exception as e:
        print(f"Error training SVC: {e}")

    #Step 4: Evaluate Classifiers 
    print("\nStep 4: Evaluating Classifiers on Validation Set")
    target_names = [f'Stage {i}' for i in range(n_clusters)]
    # Create dataframe to store validation results, start with features and true labels
    df_val_results = X_val.copy()
    df_val_results['true_stage'] = y_val
    # Add unit_number and time_in_cycles by merging based on index
    if 'unit_number' in df_class_input.columns and 'time_in_cycles' in df_class_input.columns:
        df_val_results = df_val_results.merge(
            df_class_input[['unit_number', 'time_in_cycles']],
            left_index=True, right_index=True, how='left'
        )
    else:
         print("Warning: Could not merge 'unit_number' and 'time_in_cycles' into validation results.")

    classification_accuracies = {}
    best_classifier_name = None
    best_accuracy = -1

    # Loop through each trained classifier stored in the 'classifiers' dictionary
    for name, model in classifiers.items():
        print(f"\nEvaluation for: {name.upper()}")
        y_pred = model.predict(X_val)

        # Store predictions
        df_val_results[f'predicted_stage_{name}'] = y_pred

        # Store probabilities if available
        if hasattr(model, "predict_proba"):
            try:
                y_prob = model.predict_proba(X_val)
                # Ensure y_prob has the correct shape (samples, n_clusters)
                if y_prob.shape[1] == n_clusters:
                    for i in range(n_clusters):
                        # Ensure column name uniqueness for probabilities
                        df_val_results[f'prob_stage_{i}_{name}'] = y_prob[:, i]
                else:
                     print(f"Warning: Probability array shape mismatch for {name}. Expected {n_clusters} classes, got {y_prob.shape[1]}. Skipping probability storage.")

            except Exception as e:
                 print(f"Warning: Could not get probabilities for model '{name}'. Error: {e}. Skipping probability storage.")
        else:
            print(f"Warning: predict_proba not available for model '{name}'. Skipping probability storage.")

        # 2. Classification Report & Accuracy
        print("\nClassification Report:")
        # Use labels argument in classification_report to handle cases where some stages might be missing in y_val/y_pred
        report = classification_report(y_val, y_pred, labels=range(n_clusters), target_names=target_names, zero_division=0)
        print(report)
        # Calculate and print overall accuracy
        accuracy = accuracy_score(y_val, y_pred)
        classification_accuracies[name] = accuracy
        print(f"Overall Accuracy: {accuracy:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_classifier_name = name

        # 3. Confusion Matrix Plot
        plot_confusion_matrix_heatmap(y_val, y_pred, target_names, name)

        # 4. Feature Importance / Coefficients Plotting
        plot_feature_importances(model, X_train.columns, name)

    if best_classifier_name:
        print(f"\nSelected Best Classifier (based on Validation Accuracy): {best_classifier_name.upper()} (Accuracy: {best_accuracy:.4f})")
    else:
        print("\nWarning: Could not determine the best classifier.")
        best_classifier_name = list(classifiers.keys())[0] if classifiers else None # Fallback if needed

    print(f"\nPhase 2 Evaluation Complete")
    return (classifiers, df_val_results, X_train, X_val, y_train, y_val,
            best_classifier_name, classification_accuracies, scaler_numeric)