"""
Stores configuration constants for the pipeline.
"""

#Dataset Configuration 
DATASET_PATH = 'CMaps/' # Path of the Datasets
"""
Choose the dataset or datasets to use
"""
#DATASET_IDS = ['FD001'] # Example: Dataset FD001
# DATASET_IDS = ['FD001', 'FD003'] # Example: Combine FD001 and FD003
DATASET_IDS = ['FD002', 'FD004'] # Example: Combine FD002 and FD004
# DATASET_IDS = ['FD001', 'FD002', 'FD003', 'FD004']
COMBINED_DATASET_NAME = "+".join(DATASET_IDS)

#Model Configuration 
N_CLUSTERS = 5       # Number of degradation stages (0: Normal -> 4: Failure)
RANDOM_STATE = 42

#Risk Assessment 
RISK_THRESHOLD = 0.7 # Risk Threshold for maintenance alert if the risk score is above this value

#Sensor Grouping (Informational) 
SENSOR_GROUPS = {
    'temperature': ['sensor_2', 'sensor_3', 'sensor_4'],
    'pressure': ['sensor_8', 'sensor_9', 'sensor_10', 'sensor_11'],
    'rpm': ['sensor_12', 'sensor_13'],
    'flow': ['sensor_5', 'sensor_6', 'sensor_7', 'sensor_14', 'sensor_15'],
    # Note: sensor_16 to sensor_21 are not explicitly grouped here
}
# This is just for reference and does not affect the code execution
# The sensors are grouped based on their types for better understanding of the data for the user not the model

#  Plotting 
# Number of sample engines to plot trends for (adjust as needed)
# Will try to pick samples across combined datasets if applicable
MAX_PLOT_SAMPLE_ENGINES = 5
# Sensors to attempt plotting trends for (will only plot available ones)
DEFAULT_SENSORS_TO_PLOT = ['sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_8', 'sensor_12']

#  Features 
# Window size for rolling features (if used)
ROLLING_WINDOW_SIZE = 50