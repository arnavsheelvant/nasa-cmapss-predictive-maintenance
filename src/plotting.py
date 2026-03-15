"""
Contains plotting functions for visualization throughout the pipeline.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import os

#Phase 1 Plotting

#To visualize the distribution of samples across the derived degradation stages
def plot_cluster_distribution(df, stage_col, n_clusters):
    """Distribution of samples across the derived degradation stages."""
    plt.figure(figsize=(8, 5))
    sns.countplot(x=stage_col, data=df, order=sorted(df[stage_col].unique()))
    plt.title(f'Distribution of Samples Across {n_clusters} Degradation Stages')
    plt.xlabel('Degradation Stage')
    plt.ylabel('Number of Samples')
    plt.grid(axis='y', linestyle='--')
    plt.show()

#To visualize the clusters in 2D space, we can use PCA to reduce the dimensionality of the data
def plot_pca_clusters(X_pca, labels, n_clusters):
    """Plots the clusters using PCA results"""
    plt.figure(figsize=(10, 7))
    unique_labels = sorted(np.unique(labels))
    # Ensure colors match number of actual unique labels found
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))
    for i, label in enumerate(unique_labels):
        cluster_points = X_pca[labels == label]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                    color=colors[i],
                    label=f'Stage {label}', # Label directly with the stage number
                    alpha=0.5, s=8)

    plt.title(f'Sensor Data Clusters Visualized with PCA ({n_clusters} Stages)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title="Degradation Stage", markerscale=2)
    plt.grid(True, linestyle='--')
    plt.show()

#To visualize the trends of sensors over time for a sample of engines
#This function will plot the trends of the sensors over time for a sample of engines, colored by their degradation stage
def plot_sensor_trends_by_stage(df, sensor_cols_to_plot, stage_col, sample_engines, n_clusters):
    """Plots sensor trends over time for sample engines, colored by stage."""
    print(f"\nPlotting sensor trends for sample engines: {sample_engines}")
    # Ensure df is sorted by unit_number and time_in_cycles for correct line plotting
    df_sorted = df.sort_values(by=['unit_number', 'time_in_cycles'])
    sample_df = df_sorted[df_sorted['unit_number'].isin(sample_engines)].copy()

    if sample_df.empty:
        print(f"Warning: No data found for sample engines {sample_engines}.")
        return

    num_sensors = len(sensor_cols_to_plot)
    if num_sensors == 0:
        print("Warning: No sensors provided to plot.")
        return

    num_cols = 3
    num_rows = (num_sensors + num_cols - 1) // num_cols

    plt.figure(figsize=(num_cols * 6, num_rows * 4))
    unique_stages = sorted(sample_df[stage_col].unique())
    # Ensure colors match number of unique stages present in sample
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_stages)))
    stage_color_map = {stage: colors[i] for i, stage in enumerate(unique_stages)}

    # Determine global min/max for color mapping consistency if needed
    vmin = min(unique_stages) if unique_stages else 0
    vmax = max(unique_stages) if unique_stages else n_clusters-1


    for i, sensor in enumerate(sensor_cols_to_plot):
        ax = plt.subplot(num_rows, num_cols, i + 1)

        has_data_for_sensor = False
        for engine_id in sample_engines:
            engine_data = sample_df[sample_df['unit_number'] == engine_id]
            if not engine_data.empty and sensor in engine_data.columns:
                has_data_for_sensor = True
                # Check if stage_col contains valid data before using for color
                if not engine_data[stage_col].isnull().all():
                     scatter = ax.scatter(engine_data['time_in_cycles'], engine_data[sensor],
                                         c=engine_data[stage_col], cmap='viridis', vmin=vmin, vmax=vmax,
                                         s=10, alpha=1.0, label=f'Engine {engine_id}' if i == 0 else "") # Label only once per engine
                else:
                     # Fallback if stage info is missing - plot without color mapping
                     scatter = ax.scatter(engine_data['time_in_cycles'], engine_data[sensor],
                                          s=10, alpha=1.0, label=f'Engine {engine_id}' if i == 0 else "")

                ax.plot(engine_data['time_in_cycles'], engine_data[sensor], '-', lw=0.7, alpha=0.8, color='gray')

        if not has_data_for_sensor:
            ax.text(0.5, 0.5, 'No data for this sensor', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.set_title(f'{sensor} Trend (Scaled)')
        ax.set_xlabel('Time in Cycles')
        ax.set_ylabel('Scaled Sensor Value')
        ax.grid(True, linestyle='--')

        # Create legend for stages (only add once, typically to the first plot)
        if i == 0 and unique_stages and stage_color_map:
             handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=stage_color_map[stage], markersize=8)
                        for stage in unique_stages if stage in stage_color_map]
             labels = [f'Stage {stage}' for stage in unique_stages if stage in stage_color_map]
             stage_legend = ax.legend(handles, labels, title="Degradation Stage", loc='best', fontsize='small')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle(f'Sensor Trends for Engines {sample_engines} (Colored by Stage)', fontsize=16, y=0.99)
    plt.show()

#Phase 2 Plotting

#To visualize the performance of the model on the validation set, we can plot the confusion matrix
def plot_confusion_matrix_heatmap(y_true, y_pred, stage_names, model_name):
    """Plots a heatmap of the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=stage_names,
                yticklabels=stage_names)
    plt.title(f'Confusion Matrix - {model_name.upper()} Stage Prediction')
    plt.xlabel('Predicted Stage')
    plt.ylabel('True Stage')
    plt.grid(False)
    plt.show()

#To visualize the feature importances or coefficients of the model, we can plot them as a bar chart
def plot_feature_importances(model, feature_names, model_name):
    """Plots feature importances or coefficients for a trained model."""
    print(f"\nPlotting Feature Importances/Coefficients for {model_name.upper()}...")
    importances = None
    plot_title = f'Top 20 Feature Importances/Coefficients for {model_name.upper()}'
    xlabel = 'Importance/Coefficient Value'

    if hasattr(model, 'feature_importances_'):
        importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
        xlabel = 'Importance Score'
    elif hasattr(model, 'coef_'):
        # Handle multi-class coefficients (like OvR LogReg) by averaging absolute values
        if model.coef_.shape[0] > 1:
            avg_abs_coef = np.mean(np.abs(model.coef_), axis=0)
            importances = pd.Series(avg_abs_coef, index=feature_names).sort_values(ascending=False)
            xlabel = 'Average Absolute Coefficient Value'
        # Handle single output coefficients (like linear kernel SVC with 2 classes, or binary LogReg)
        elif model.coef_.shape[0] == 1:
             importances_raw = pd.Series(model.coef_[0], index=feature_names)
             # Sort by absolute value but keep original sign for plotting
             sorted_indices = importances_raw.abs().sort_values(ascending=False).index
             importances = importances_raw[sorted_indices]
             xlabel = 'Coefficient Value'
        else:
            print(f"Coefficient array shape {model.coef_.shape} not directly plottable for importance.")
            return # Cannot plot
    else:
        print(f"Feature importances or coefficients are not directly available for model type: {type(model).__name__}")
        return # Cannot plot

    if importances is not None and not importances.empty:
        # Plot top 20 features
        importances_to_plot = importances.head(20)
        plt.figure(figsize=(10, max(6, len(importances_to_plot) // 2)))
        sns.barplot(x=importances_to_plot.values, y=importances_to_plot.index)
        plt.title(plot_title)
        plt.xlabel(xlabel)
        plt.ylabel('Features')
        plt.tight_layout()
        plt.show()
    elif importances is not None and importances.empty:
         print("Warning: Importance Series was empty.")

#Phase 3 Plotting

#To visualize the true vs predicted TTNS (Time-to-Next-Stage) values, we can plot them in a scatter plot
#This will help us understand how well the model is predicting the time-to-next-stage for each engine
def plot_ttns_scatter(y_true, y_pred, model_name):
    """Plots a scatter plot of true vs predicted TTNS."""
    plt.figure(figsize=(8, 8))
    # Use only valid (non-NaN) points for plotting range and ideal line
    valid_indices = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true_valid = y_true[valid_indices]
    y_pred_valid = y_pred[valid_indices]

    if len(y_true_valid) == 0:
        print(f"Warning: No valid TTNS data points to plot for {model_name}.")
        plt.text(0.5, 0.5, 'No valid data points', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
        plt.title(f'True vs. Predicted TTNS - {model_name.upper()} (No Data)')
        plt.xlabel('True TTNS (Cycles)')
        plt.ylabel('Predicted TTNS (Cycles)')
        plt.show()
        return

    min_val = min(y_true_valid.min(), y_pred_valid.min()) if len(y_true_valid)>0 else 0
    max_val = max(y_true_valid.max(), y_pred_valid.max()) if len(y_true_valid)>0 else 100 # Default max if empty

    plt.scatter(y_true_valid, y_pred_valid, alpha=0.3, s=10, label='Predictions')
    plt.plot([min_val, max_val], [min_val, max_val], '--r', linewidth=2, label='Ideal Fit')
    plt.title(f'True vs. Predicted Time-to-Next-Stage (TTNS) for {model_name.upper()}')
    plt.xlabel('True TTNS (Cycles)')
    plt.ylabel('Predicted TTNS (Cycles)')
    plt.legend()
    plt.grid(True, linestyle='--')
    # Set limits slightly beyond the data range
    plt.xlim([min_val - (max_val-min_val)*0.05, max_val + (max_val-min_val)*0.05])
    plt.ylim([min_val - (max_val-min_val)*0.05, max_val + (max_val-min_val)*0.05])
    # Ensure non-negative limits if data starts near 0
    if min_val >= 0:
        plt.xlim(left=max(-0.5, plt.xlim()[0]))
        plt.ylim(bottom=max(-0.5, plt.ylim()[0]))

    plt.show()


#Phase 4 Plotting

#To visualize the risk score and true stage over time for a single engine, we can plot them on the same graph
#This will help us understand how the risk score changes over time and how it relates to the true degradation stage of the engine
def plot_risk_trend_single_engine(engine_data, engine_id, n_clusters, risk_threshold, ax1):
    """Plots risk score and true stage for a single engine on given axes."""
    if not engine_data.empty:
        # Plot Normalized Risk Score on primary axis
        color = 'tab:red'
        ax1.set_xlabel('Time in Cycles')
        ax1.set_ylabel('Normalized Risk Score', color=color)
        ax1.plot(engine_data['time_in_cycles'], engine_data['normalized_risk_score'], color=color, label='Norm. Risk Score', marker='.', markersize=3)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.axhline(y=risk_threshold, color=color, linestyle='--', linewidth=1.5, label=f'Alert Threshold ({risk_threshold})')
        ax1.set_ylim(-0.05, 1.05)
        ax1.grid(True, axis='y', linestyle=':')

        # Plot True Degradation Stage on secondary y-axis
        ax2 = ax1.twinx() # instantiate a second axes that shares the same x-axis
        color = 'tab:blue'
        ax2.set_ylabel('True Degradation Stage', color=color)
         # Check if 'true_stage' exists and is not all NaN
        if 'true_stage' in engine_data.columns and not engine_data['true_stage'].isnull().all():
             ax2.step(engine_data['time_in_cycles'], engine_data['true_stage'], where='post', color=color, alpha=0.6, linestyle='--', linewidth=1.5, label='True Stage')
             ax2.tick_params(axis='y', labelcolor=color)
             ax2.set_yticks(range(n_clusters))
             ax2.set_yticklabels([f'Stage {j}' for j in range(n_clusters)])
             ax2.set_ylim(-0.5, n_clusters - 0.5)
        else:
             ax2.text(0.5, 0.5, 'True Stage Missing', horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes, color=color)
        ax1.set_title(f'Engine {engine_id}: Normalized Risk Score and True Stage vs. Time')
        # Combine legends
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper left')
    else:
        ax1.text(0.5, 0.5, f'Engine {engine_id} not found in validation set',
                horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
        ax1.set_title(f'Engine {engine_id}: No Data')

# To visualize the risk score trends for a sample of engines, we can plot them in a grid layout
#This will help us understand how the risk score changes over time for different engines
def plot_risk_trends_sample_engines(df_val_results, sample_engines, n_clusters, risk_threshold):
     """Plots risk score trends for a list of sample engines."""
     print("\nPlotting Normalized Risk Score trends for sample engines...")
     if not sample_engines:
         print("No sample engines provided for plotting.")
         return

     df_val_results_sorted = df_val_results.sort_values(by=['unit_number', 'time_in_cycles'])
     plt.figure(figsize=(15, len(sample_engines) * 4.5))

     for i, engine_id in enumerate(sample_engines):
         engine_data = df_val_results_sorted[df_val_results_sorted['unit_number'] == engine_id]
         ax1 = plt.subplot(len(sample_engines), 1, i + 1)
         plot_risk_trend_single_engine(engine_data, engine_id, n_clusters, risk_threshold, ax1)

     plt.tight_layout()
     plt.show()

# To visualize the risk score trends for all engines in the validation set, and save them to a directory
def plot_and_save_engine_risk_trends(df_val_results, output_plot_dir, combined_dataset_name, n_clusters, risk_threshold):
    """Plots risk trends for all engines in the validation set and saves them."""
    print(f"\nPlotting and Saving Individual Engine Risk Trends")
    output_dir_path = f"{output_plot_dir}_{combined_dataset_name}"

    try:
        os.makedirs(output_dir_path, exist_ok=True) # Create directory if it doesn't exist
        print(f"Plots will be saved in directory: '{output_dir_path}'")
    except OSError as error:
        print(f"Error creating directory {output_dir_path}: {error}")
        return # Stop if directory cannot be created

    # Ensure data is sorted for plotting lines correctly
    df_val_results_sorted = df_val_results.sort_values(by=['unit_number', 'time_in_cycles'])

    # Get list of unique engines in the validation set
    engines_in_validation = df_val_results_sorted['unit_number'].unique()
    print(f"Found {len(engines_in_validation)} unique engines in the validation set.")

    # Loop through each engine and create/save a plot
    plot_count = 0
    for engine_id in engines_in_validation:
        engine_data = df_val_results_sorted[df_val_results_sorted['unit_number'] == engine_id]
        if not engine_data.empty:
            fig, ax1 = plt.subplots(figsize=(12, 6))
            plot_risk_trend_single_engine(engine_data, engine_id, n_clusters, risk_threshold, ax1)
            safe_engine_id = str(engine_id).replace('/', '_').replace('\\', '_')
            plot_filename = f"engine_{safe_engine_id}_risk_trend.png"
            plot_filepath = os.path.join(output_dir_path, plot_filename)
            try:
                plt.savefig(plot_filepath)
                plot_count += 1
            except Exception as e:
                print(f"Warning: Could not save plot for engine {engine_id} to {plot_filepath}. Error: {e}")
            plt.close(fig) # Close the figure to free memory
    print(f"\nFinished plotting. Saved {plot_count} plots to '{output_dir_path}'.")