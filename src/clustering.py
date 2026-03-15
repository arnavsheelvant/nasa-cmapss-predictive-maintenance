"""
Performs sensor scaling, clustering (KMeans, GMM), and interprets clusters
into degradation stages based on time progression.
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import time

# From the set of sensors, we drop the ones with low variance and then scale the remaining ones.
# This is important for clustering as it helps to ensure that all features contribute equally to the distance calculations.
def identify_and_scale_sensors(df_input_with_op_cond, sensor_cols_initial, std_threshold=0.01, op_cond_col='op_condition_id'):
    """
    Identifies low variance sensors, drops them, and scales the remaining ones.

    Args:
        df_train (pd.DataFrame): Input dataframe with sensor columns.
        sensor_cols (list): List of sensor column names.
        std_threshold (float): Standard deviation threshold to identify low variance.

    Returns:
        tuple: (pd.DataFrame, StandardScaler, list)
               - df_processed: Dataframe with low variance sensors dropped and others scaled.
               - scaler_sensors: Fitted StandardScaler object for sensors.
               - active_sensor_cols: List of sensor columns used after dropping low variance ones.
    """
    print("\nSensor Preprocessing (with Condition-Specific Scaling)")
    df_processed = df_input_with_op_cond.copy()
    print("Checking for low-variance sensors (globally)...")
    current_sensor_cols = [col for col in sensor_cols_initial if col in df_processed.columns]
    if not current_sensor_cols:
        print("Error: No sensor columns found in df_processed to check for low variance.")
        # Return early with an empty list for active_sensor_cols if no sensors
        return df_processed, []
    
    sensor_data_for_std_check = df_processed[current_sensor_cols].apply(pd.to_numeric, errors='coerce')
    sensor_std = sensor_data_for_std_check.std().sort_values()

    low_variance_sensors = sensor_std[sensor_std < std_threshold].index.tolist()

    if low_variance_sensors:
        print(f"Dropping Sensors columns with std dev < {std_threshold}: {low_variance_sensors}")
        df_processed.drop(columns=low_variance_sensors, inplace=True)
        active_sensor_cols = [col for col in current_sensor_cols if col not in low_variance_sensors]
    else:
        print(f"-> No global low-variance sensors found with std dev < {std_threshold}. No sensors dropped based on global std.")
        active_sensor_cols = current_sensor_cols[:]

    if not active_sensor_cols:
        print("Error: No active sensor columns remaining after low-variance check. Cannot proceed with scaling.")
        return df_processed, []
    print(f"Selected {len(active_sensor_cols)} active sensor columns for condition-specific scaling: {active_sensor_cols[:5]}...")

    # Condition-Specific Scaling for Active Sensor Data
    if op_cond_col not in df_processed.columns:
        print(f"Warning: Operating condition column '{op_cond_col}' not found. Performing GLOBAL scaling instead.")
        # Fallback to global scaling if op_cond_col is missing
        scaler_global_sensors = StandardScaler()
        df_processed[active_sensor_cols] = scaler_global_sensors.fit_transform(df_processed[active_sensor_cols])
        print("Active sensor data scaled GLOBALLY as fallback.")
        # In this fallback, a scaler object could be returned, but to keep signature consistent:
        return df_processed, active_sensor_cols
    
    print(f"\nScaling {len(active_sensor_cols)} active sensor data *within each operating condition* using StandardScaler (op_cond_col: '{op_cond_col}')...")
    scaled_sensor_data_list = []
    cols_to_scale_in_groups = [col for col in active_sensor_cols if col in df_processed.columns]

    if not cols_to_scale_in_groups:
        print("Warning: No active sensor columns found in dataframe groups to scale.")
        return df_processed, active_sensor_cols
    
    for condition_id, group_df in df_processed.groupby(op_cond_col):
        group_copy = group_df.copy()
        scaler_sensors_condition = StandardScaler()
        # Ensure group_df[cols_to_scale_in_groups] is not empty
        if not group_df[cols_to_scale_in_groups].empty:
            group_copy[cols_to_scale_in_groups] = scaler_sensors_condition.fit_transform(group_df[cols_to_scale_in_groups])
        else:
            print(f"Warning: Group for condition_id {condition_id} has no data for specified sensor columns.")
        scaled_sensor_data_list.append(group_copy)

    if scaled_sensor_data_list:
        df_processed = pd.concat(scaled_sensor_data_list).sort_index()
        print("Active sensor data scaled within each operating condition.")
    else:
        print("Warning: No data was scaled; scaled_sensor_data_list is empty.")

    print("Sample of processed data (sensor values are now condition-scaled if op_cond_col was present):")

    display_cols = ['unit_number', 'time_in_cycles']
    if op_cond_col in df_processed.columns:
        display_cols.append(op_cond_col)
    display_cols.extend(active_sensor_cols[:3]) # Show first 3 active sensors
    display_cols = [col for col in display_cols if col in df_processed.columns] # Ensure all display_cols exist
    print(df_processed[display_cols].head())

    return df_processed, active_sensor_cols

# This function interprets the cluster labels based on the average time_in_cycles
# for each cluster. It maps the clusters to degradation stages (0 to N-1) based on the average time_in_cycles.
def interpret_clusters_by_time(df, cluster_col_name, stage_col_name):
    """
    Interprets cluster labels based on average time_in_cycles and maps them to stages.

    Args:
        df (pd.DataFrame): Dataframe with cluster labels and time_in_cycles.
        cluster_col_name (str): Name of the column containing raw cluster labels.
        stage_col_name (str): Name of the new column to store derived stages.

    Returns:
        tuple: (pd.DataFrame, dict)
               - df: Dataframe with the new stage column added.
               - stage_map: Dictionary mapping raw cluster label to degradation stage.
    """
    print(f"\nInterpreting clusters '{cluster_col_name}' based on average time_in_cycles...")
    # Ensure cluster labels are numeric for grouping
    df[cluster_col_name] = pd.to_numeric(df[cluster_col_name], errors='coerce')
    df_valid_clusters = df.dropna(subset=[cluster_col_name])

    if df_valid_clusters.empty:
        print(f"Warning: No valid cluster labels found in column '{cluster_col_name}'. Cannot interpret stages.")
        df[stage_col_name] = np.nan # Assign NaN stage if interpretation fails
        return df, {}

    cluster_avg_time = df_valid_clusters.groupby(cluster_col_name)['time_in_cycles'].mean().sort_values()

    print(f"\nAverage 'time_in_cycles' for each raw cluster label in '{cluster_col_name}' (sorted):")
    print(cluster_avg_time)

    # Now we map these clusters to the degradation stages (0 to N-1)
    stage_map = {cluster_label: stage for stage, cluster_label in enumerate(cluster_avg_time.index)}

    print(f"\nMapping from raw cluster label ({cluster_col_name}) to degradation stage ({stage_col_name}):")
    map_df = pd.DataFrame(list(stage_map.items()), columns=["Cluster Number", "Degradation Stage"])
    print(map_df)

    # Apply the mapping
    df[stage_col_name] = df[cluster_col_name].map(stage_map)
    # Handle any potential NaNs introduced if some cluster labels weren't in the map (shouldn't happen with groupby)
    df[stage_col_name] = df[stage_col_name].fillna(-1) # Or some other indicator
    return df, stage_map

# This function runs the clustering phase, applying KMeans and GMM clustering,
def run_clustering_phase(df_train_scaled, active_sensor_cols, n_clusters, random_state):
    """
    Applies KMeans and GMM clustering, interprets stages, and performs PCA.

    Args:
        df_train_scaled (pd.DataFrame): Dataframe with scaled active sensor data.
        active_sensor_cols (list): List of scaled sensor columns to use for clustering.
        n_clusters (int): Number of clusters/stages.
        random_state (int): Random state for reproducibility.

    Returns:
        tuple: Contains:
            - df_clustered (pd.DataFrame): Dataframe with cluster labels and derived stages.
            - kmeans_model (KMeans): Fitted KMeans model.
            - gmm_model (GaussianMixture): Fitted GMM model.
            - kmeans_stage_map (dict): Mapping for KMeans stages.
            - gmm_stage_map (dict): Mapping for GMM stages.
            - pca (PCA): Fitted PCA object.
            - X_pca (np.ndarray): PCA-transformed sensor data.
    """
    print("\nPhase 1: Clustering")
    df_clustered = df_train_scaled.copy()

    if not active_sensor_cols or not any(col in df_clustered.columns for col in active_sensor_cols):
        print("Error: No active sensor columns available for clustering. Skipping clustering phase.")
        # Ensure all expected return values are provided even in failure cases
        dummy_kmeans = KMeans(n_clusters=n_clusters if n_clusters > 0 else 1, random_state=random_state, n_init=1)
        dummy_gmm = GaussianMixture(n_components=n_clusters if n_clusters > 0 else 1, random_state=random_state, n_init=1)
        dummy_pca = PCA(n_components=min(2, len(active_sensor_cols) if active_sensor_cols else 1)) # Handle n_components
        # Create necessary columns if they don't exist
        for col in ['kmeans_cluster_label', 'degradation_stage', 'gmm_cluster_label', 'gmm_stage']:
            if col not in df_clustered.columns:
                df_clustered[col] = np.nan
        X_pca_dummy = np.empty((len(df_clustered), min(2, len(active_sensor_cols) if active_sensor_cols else 1))) * np.nan
        return df_clustered, dummy_kmeans, dummy_gmm, {}, {}, dummy_pca, X_pca_dummy
    
    clustering_features = df_clustered[active_sensor_cols]

    #KMeans Clustering
    print(f"\nApplying KMeans clustering with k={n_clusters} on scaled sensor data...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10, algorithm='lloyd')
    start_time = time.time()
    df_clustered['kmeans_cluster_label'] = kmeans.fit_predict(clustering_features)
    end_time = time.time()
    print(f"KMeans fitting completed in {end_time - start_time:.2f} seconds.")
    print("Distribution of raw KMeans cluster labels:")
    print(df_clustered['kmeans_cluster_label'].value_counts().sort_index())

    # Interpret KMeans clusters
    df_clustered, kmeans_stage_map = interpret_clusters_by_time(
        df_clustered, 'kmeans_cluster_label', 'degradation_stage' # Use 'degradation_stage' as the primary derived stage column
    )

    #Gaussian Mixture Model (GMM) Clustering
    print(f"\nApplying Gaussian Mixture Model (n_components={n_clusters})...")
    start_time = time.time()
    gmm = GaussianMixture(n_components=n_clusters,
                          covariance_type='full',
                          random_state=random_state,
                          n_init=5,
                          max_iter=150
                         )
    df_clustered['gmm_cluster_label'] = gmm.fit_predict(clustering_features)
    end_time = time.time()
    print(f"GMM fitting completed in {end_time - start_time:.2f} seconds.")
    print("Distribution of raw GMM cluster labels:")
    print(df_clustered['gmm_cluster_label'].value_counts().sort_index())

    # Interpret GMM clusters
    df_clustered, gmm_stage_map = interpret_clusters_by_time(
        df_clustered, 'gmm_cluster_label', 'gmm_stage'
    )
    print("GMM Stage Mapping (Cluster -> Stage):", gmm_stage_map)

    #PCA for Visualization
    print("\nPerforming PCA for visualization...")
    pca = PCA(n_components=2, random_state=random_state) # Reduce to 2 components for visualization, so that we can plot the clusters on a 2D plane
    X_pca = pca.fit_transform(clustering_features)
    print(f"PCA complete. Explained variance ratio by 2 components: {pca.explained_variance_ratio_.sum():.4f}")

    print("\nPhase 1 Complete")
    print("DataFrame now includes scaled sensors, cluster labels, and derived stages.")
    return df_clustered, kmeans, gmm, kmeans_stage_map, gmm_stage_map, pca, X_pca