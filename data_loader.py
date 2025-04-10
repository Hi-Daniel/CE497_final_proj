import os
import glob
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm # For progress bars in notebook

def parse_vector(vector_str):
    """Parses a vector string like '(x, y, z)' into a list of floats."""
    try:
        return [float(x.strip()) for x in vector_str.strip('()').split(',')]
    except:
        return [np.nan, np.nan, np.nan] # Handle potential parsing errors

def parse_xml_file(file_path):
    """Parses a single XML data file into a dictionary."""

    tree = ET.parse(file_path)
    root = tree.getroot()
    
    # Get display dimensions and start time
    display_dims = [int(root.find("./GazeData/DisplayDimensions").get(val)) 
                    for val in ["Width", "Height"]]
    gaze_start_time = int(root.find("./GazeData/Timestamp").text)
    cam_start_time = int(root.find("./CameraData/Timestamp").text)

    X_2d = []
    gaze_t = []
    pupil = []
    gaze_origin = []
    gaze_direction = []
    cam_D = []
    cam_X = []
    cam_t = []
    # Parse all the data arrays using the example code approach    
    for p in root.findall("GazeData"):
        gaze_t.append((int(p.find('Timestamp').text) - gaze_start_time)/1e6)
        pupil.append(float(p.find('pupil').get('average_pupildiameter')) if p.find('pupil') is not None else np.nan)
        #check if the element has a PositionOnDisplayArea child
        if p.find("PositionOnDisplayArea") is not None:
            X_2d.append([float(p.find("PositionOnDisplayArea").get('X')), float(p.find("PositionOnDisplayArea").get('Y'))])
        else:
            X_2d.append([np.nan, np.nan])

        #get gaze origin
        if p.find("GazeOrigin") is not None:
            #get the combined gaze ray screen child
            gaze_origin.append(parse_vector(p.find("GazeOrigin").find("CombinedGazeRayScreen").get('Origin')))
            gaze_direction.append(parse_vector(p.find("GazeOrigin").find("CombinedGazeRayScreen").get('Direction')))
        else:
            gaze_origin.append([np.nan, np.nan, np.nan])
            gaze_direction.append([np.nan, np.nan, np.nan])
        
    cam_t = [(int(p.find('Timestamp').text) - cam_start_time)/1e6 for p in root.findall("CameraData")]
    cam_D = [[float(p.get("X")), float(p.get("Y")), float(p.get("Z"))] for p in root.findall("CameraData/CameraDirection")]
    cam_X = [[float(p.get("X")), float(p.get("Y")), float(p.get("Z"))] for p in root.findall("CameraData/CameraOrigin")]

    #convert to numpy arrays    
    gaze_t = np.array(gaze_t)
    X_2d = np.array(X_2d)
    pupil = np.array(pupil)
    gaze_origin = np.array(gaze_origin)
    gaze_direction = np.array(gaze_direction)
    cam_D = np.array(cam_D)
    cam_X = np.array(cam_X)
    cam_t = np.array(cam_t)
    
    data = {
        'Time_sec': gaze_t,
        'X_2d_X': X_2d[:,0], 'X_2d_Y': X_2d[:,1],
        'PupilDiameter': pupil,
        'GazeOrigin_X': gaze_origin[:,0], 'GazeOrigin_Y': gaze_origin[:,1], 'GazeOrigin_Z': gaze_origin[:,2],
        'GazeDirection_X': gaze_direction[:,0], 'GazeDirection_Y': gaze_direction[:,1], 'GazeDirection_Z': gaze_direction[:,2],
        'CameraDirection_X': cam_D[:,0], 'CameraDirection_Y': cam_D[:,1], 'CameraDirection_Z': cam_D[:,2],
        'CameraOrigin_X': cam_X[:,0], 'CameraOrigin_Y': cam_X[:,1], 'CameraOrigin_Z': cam_X[:,2],
        'CameraTime_sec': cam_t
    }
    
    return data

def data_dict_to_df(data_dict):
    """Converts a dictionary of data into a pandas DataFrame."""
    cam_df = pd.DataFrame({'Time_sec': data_dict['CameraTime_sec'],
                           'CameraDirection_X': data_dict['CameraDirection_X'],
                           'CameraDirection_Y': data_dict['CameraDirection_Y'],
                           'CameraDirection_Z': data_dict['CameraDirection_Z'],
                           'CameraOrigin_X': data_dict['CameraOrigin_X'],
                           'CameraOrigin_Y': data_dict['CameraOrigin_Y'],
                           'CameraOrigin_Z': data_dict['CameraOrigin_Z']})
    gaze_df = pd.DataFrame({'Time_sec': data_dict['Time_sec'],
                           'X_2d_X': data_dict['X_2d_X'],
                           'X_2d_Y': data_dict['X_2d_Y'],
                           'PupilDiameter': data_dict['PupilDiameter'],
                           'GazeOrigin_X': data_dict['GazeOrigin_X'],
                           'GazeOrigin_Y': data_dict['GazeOrigin_Y'],
                           'GazeOrigin_Z': data_dict['GazeOrigin_Z'],
                           'GazeDirection_X': data_dict['GazeDirection_X'],
                           'GazeDirection_Y': data_dict['GazeDirection_Y'],
                           'GazeDirection_Z': data_dict['GazeDirection_Z']})
    
    #merge the two dataframes on the 'Time_sec' column
    df = pd.merge(gaze_df, cam_df, on='Time_sec', how='outer')
    return df
    
def load_all_data(data_folder="test_1"):
    """Loads all XML files from the specified folder."""
    xml_files = glob.glob(os.path.join(data_folder, "*.xml"))
    all_data = {}
    print(f"Found {len(xml_files)} XML files in {data_folder}")
    for file_path in tqdm(xml_files, desc="Loading XML files"):
        file_name = os.path.basename(file_path)
        data = parse_xml_file(file_path)
        if data is not None:
            # Extract user ID and task type from filename (adjust pattern if needed)
            parts = file_name.replace('.xml', '').split('_')
            user_id = parts[0] # Assuming user ID is the first part
            task = parts[-1]   # Assuming task is the last part ('truss' or 'warehouse')
            key = f"{user_id}_{task}"
            all_data[key] = data
            print(f"Loaded {file_name} ({len(data)} rows) as key '{key}'")
        else:
            print(f"Skipped {file_name} due to parsing errors or empty data.")

    # Separate truss and warehouse data
    truss_data = {k: v for k, v in all_data.items() if k.endswith('_truss')}
    warehouse_data = {k: v for k, v in all_data.items() if k.endswith('_warehouse')}

    print(f"\nLoaded {len(truss_data)} truss datasets.")
    print(f"Loaded {len(warehouse_data)} warehouse datasets.")

    return truss_data, warehouse_data

def create_windows(df, window_size_points, window_overlap):
    """
    Breaks a DataFrame into overlapping windows.
    
    Args:
        df: pandas DataFrame to window
        window_size_points: number of points in each window
        window_overlap: overlap between windows as percentage (0-1)
    """
    windows = []
    step_size = int(window_size_points * (1 - window_overlap))
    for i in range(0, len(df) - window_size_points + 1, step_size):
        windows.append(df.iloc[i:i + window_size_points])
    return windows

import numpy as np
import pandas as pd

def fill_dataframe(df, zero_sensitive_cols=None):
    """
    Cleans a DataFrame by replacing bad zeros with NaNs and interpolating missing values.
    Args:
        df (pd.DataFrame): Raw input DataFrame.
        zero_sensitive_cols (list): Columns where 0 is considered missing.
    Returns:
        pd.DataFrame: Cleaned and interpolated DataFrame.
    """
    df = df.copy()

    # Replace bad zeros with NaN only in specific columns
    if zero_sensitive_cols:
        df[zero_sensitive_cols] = df[zero_sensitive_cols].replace(0, np.nan)

    # Interpolate everything linearly
    df = df.interpolate(method='linear', limit_direction='both')

    return df


def anonymize_window(window_df):
    """
    Anonymizes a single window by making positions and orientations relative.
    Focuses on velocity, acceleration, and relative changes rather than absolute coords.
    """
    if window_df.empty or len(window_df) < 2:
        return window_df # Cannot process empty or single-row windows

    anonymized = window_df.copy()

    # 1. Anonymize Camera Position/Direction: zero mean position and direction, normalized direction
    anonymized['CameraOrigin_X'] = anonymized['CameraOrigin_X'] - anonymized['CameraOrigin_X'].mean()
    anonymized['CameraOrigin_Y'] = anonymized['CameraOrigin_Y'] - anonymized['CameraOrigin_Y'].mean()
    anonymized['CameraOrigin_Z'] = anonymized['CameraOrigin_Z'] - anonymized['CameraOrigin_Z'].mean()
    anonymized['CameraDirection_X'] = anonymized['CameraDirection_X'] - anonymized['CameraDirection_X'].mean()
    anonymized['CameraDirection_Y'] = anonymized['CameraDirection_Y'] - anonymized['CameraDirection_Y'].mean()
    anonymized['CameraDirection_Z'] = anonymized['CameraDirection_Z'] - anonymized['CameraDirection_Z'].mean()

    # 2. Anonymize Gaze Position/Direction: Use zero mean origin and normalized direction
    anonymized['GazeOrigin_X'] = anonymized['GazeOrigin_X'] - anonymized['GazeOrigin_X'].mean()
    anonymized['GazeOrigin_Y'] = anonymized['GazeOrigin_Y'] - anonymized['GazeOrigin_Y'].mean()
    anonymized['GazeOrigin_Z'] = anonymized['GazeOrigin_Z'] - anonymized['GazeOrigin_Z'].mean()
    return anonymized