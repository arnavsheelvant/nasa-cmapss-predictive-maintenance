"""
Handles loading and initial preparation of the CMAPSS data.
"""
import pandas as pd
import os

def load_and_preprocess_raw_data(dataset_path, dataset_ids):
    """
    Loads specified CMAPSS training datasets, combines them,
    creates unique unit numbers, and optionally filters to common columns.

    Args:
        dataset_path (str): Path to the directory containing dataset files.
        dataset_ids (list): List of dataset IDs (e.g., ['FD001', 'FD003']).
        
    Returns:
        pandas.DataFrame or None: Combined dataframe with unique unit numbers,
                                  or None if no data could be loaded.
    """
    # Define column names (ensure these are generally applicable or adjust if needed)
    columns = ['unit_number', 'time_in_cycles'] + \
              [f'operational_setting_{i}' for i in range(1, 4)] + \
              [f'sensor_{i}' for i in range(1, 22)]

    all_dfs = []
    print(f"\nLoading Data for Datasets: {dataset_ids}")
    for fd_id in dataset_ids:   #Loop through the dataset ids
        filepath = os.path.join(dataset_path, f"train_{fd_id}.txt")
        print(f"Attempting to load: {filepath}")
        try:
            df = pd.read_csv( #load the dataset
                filepath,
                sep=r'\s+',
                header=None,
                names=columns,
                engine='python',
                on_bad_lines='warn'
            )
            df['dataset_id'] = fd_id # Add dataset identifier

            #since unit_number is not unique across datasets, create a unique identifier
            # by combining dataset_id and unit_number
            df['unit_number'] = df['dataset_id'] + '_' + df['unit_number'].astype(str)

            print(f"Successfully loaded train_{fd_id}.txt. Shape: {df.shape}")
            all_dfs.append(df)
        #Error handling for file not found or other issues these wont occur but its good to have
        except FileNotFoundError: # Handle file not found error
            print(f"Error: File not found at '{filepath}'.")
            print(f"Warning: Skipping dataset {fd_id}.")
        except Exception as e: # Handle other potential errors
            print(f"An error occurred loading {filepath}: {e}")

    if not all_dfs:
         print("Error: No data was loaded successfully.")
         return None

    # Concatenate all loaded dataframes
    df_train_raw = pd.concat(all_dfs, ignore_index=True)
    print(f"\nCombined raw data shape: {df_train_raw.shape}")
    print("Sample of combined data (note unique 'unit_number'):")
    print(df_train_raw.head())

    #Optional but Recommended: Filter to Common Columns
    #Use this especially if combining across FD001/3 and FD002/4
    if len(dataset_ids) > 1:
         print("\nIdentifying common columns across loaded datasets...")
         # Get column sets, handle potential loading errors if a df is None
         col_sets = [set(df.columns) for df in all_dfs if df is not None]
         if not col_sets:
             print("Error: Could not determine columns.")
             return None
         common_cols_set = col_sets[0].intersection(*col_sets[1:])
         # Define desired order (superset of potential columns)
         desired_order = ['unit_number', 'time_in_cycles', 'dataset_id'] + \
                         [f'operational_setting_{i}' for i in range(1, 4)] + \
                         [f'sensor_{i}' for i in range(1, 22)]
         final_ordered_common_cols = [col for col in desired_order if col in common_cols_set]

         if not final_ordered_common_cols:
             print("Error: No common columns found after intersection.")
             return None

         print(f"Found {len(final_ordered_common_cols)} common columns. Processing using these in defined order.")
         # Ensure essential columns are kept if somehow dropped
         if 'unit_number' not in final_ordered_common_cols: final_ordered_common_cols.insert(0,'unit_number')
         if 'time_in_cycles' not in final_ordered_common_cols: final_ordered_common_cols.insert(1,'time_in_cycles')
         if 'dataset_id' not in final_ordered_common_cols and 'dataset_id' in df_train_raw.columns: final_ordered_common_cols.insert(2,'dataset_id')
         final_ordered_common_cols = list(dict.fromkeys(final_ordered_common_cols)) # Remove duplicates
         df_train_raw = df_train_raw[final_ordered_common_cols]
         print(f"Data filtered to common columns. New shape: {df_train_raw.shape}")

    #Final Checks
    print("\nIdentifying final feature columns...")
    all_available_columns = df_train_raw.columns
    sensor_cols = [col for col in all_available_columns if col.startswith('sensor_')]
    op_setting_cols = [col for col in all_available_columns if col.startswith('operational_setting_')]
    print(f"Using {len(sensor_cols)} sensor columns and {len(op_setting_cols)} operational setting columns.")
    # Check if essential columns exist
    if not all(col in df_train_raw.columns for col in ['unit_number', 'time_in_cycles']):
         print("Error: Essential columns ('unit_number', 'time_in_cycles') missing after loading/filtering.")
         return None # Or raise exception
    print(f"Data loading and initial prep complete. Final shape: {df_train_raw.shape}")
    return df_train_raw