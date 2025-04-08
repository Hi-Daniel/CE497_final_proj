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
    """Parses a single XML data file into a pandas DataFrame."""
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Error parsing {file_path}: {e}")
        return None

    data = []
    # The data seems to be a sequence of <Data> elements,
    # let's assume the file contains many such blocks directly under a root element (or maybe just one)
    # Adjusting based on the example which seems to have <Data> -> <GazeData>, <CameraData> pairs
    # If the actual files have multiple <Data> blocks sequentially, the logic might need adjustment.

    # Assuming the root contains pairs of GazeData and CameraData, possibly interleaved or grouped.
    # Let's find all GazeData and assume CameraData follows or is linked by Timestamp.
    # A more robust approach might be needed if the structure varies significantly.

    gaze_entries = root.findall('.//GazeData')
    camera_entries = root.findall('.//CameraData')

    # Create dictionaries keyed by timestamp for easy merging
    gaze_dict = {}
    for gaze in gaze_entries:
        ts = gaze.findtext('Timestamp')
        if not ts: continue
        combined_gaze = gaze.find('.//CombinedGazeRayScreen')
        pupil = gaze.find('.//pupil')
        intersection = gaze.find('.//IntersectionPoint')
        hit_object = gaze.find('.//HitObject')
        pos_display = gaze.find('.//PositionOnDisplayArea')

        gaze_dict[ts] = {
            'Timestamp': int(ts),
            'GazeOrigin': parse_vector(combined_gaze.get('Origin')) if combined_gaze is not None else [np.nan]*3,
            'GazeDirection': parse_vector(combined_gaze.get('Direction')) if combined_gaze is not None else [np.nan]*3,
            'GazeValid': combined_gaze.get('Valid') == 'True' if combined_gaze is not None else False,
            'AvgPupilDiameter': float(pupil.get('average_pupildiameter')) if pupil is not None and pupil.get('average_pupildiameter') else np.nan,
            'IntersectionPoint': [float(intersection.get(ax)) if intersection is not None else np.nan for ax in ['X', 'Y', 'Z']],
            'HitObjectName': hit_object.get('Name') if hit_object is not None else None,
            'ObjectPosition': [float(hit_object.find('.//ObjectPosition').get(ax)) if hit_object is not None and hit_object.find('.//ObjectPosition') is not None else np.nan for ax in ['X', 'Y', 'Z']],
            'PositionOnDisplayX': float(pos_display.get('X')) if pos_display is not None else np.nan,
            'PositionOnDisplayY': float(pos_display.get('Y')) if pos_display is not None else np.nan,
        }

    camera_dict = {}
    for camera in camera_entries:
        ts = camera.findtext('Timestamp')
        if not ts: continue
        camera_dict[ts] = {
            'CameraOrigin': [float(camera.find('.//CameraOrigin').get(ax)) if camera.find('.//CameraOrigin') is not None else np.nan for ax in ['X', 'Y', 'Z']],
            'CameraDirection': [float(camera.find('.//CameraDirection').get(ax)) if camera.find('.//CameraDirection') is not None else np.nan for ax in ['X', 'Y', 'Z']],
        }

    # Merge gaze and camera data based on timestamp
    merged_data = []
    all_timestamps = sorted(list(set(gaze_dict.keys()) | set(camera_dict.keys())))

    for ts in all_timestamps:
        entry = {'Timestamp_str': ts} # Keep original string TS if needed later
        gaze_data = gaze_dict.get(ts, {})
        camera_data = camera_dict.get(ts, {})
        entry.update(gaze_data)
        entry.update(camera_data)
        # Ensure essential keys exist even if data was missing
        entry.setdefault('Timestamp', int(ts) if ts.isdigit() else np.nan)
        merged_data.append(entry)

    if not merged_data:
        print(f"Warning: No valid data extracted from {file_path}")
        return None

    df = pd.DataFrame(merged_data)

    # Expand vector columns
    for col in ['GazeOrigin', 'GazeDirection', 'IntersectionPoint', 'ObjectPosition', 'CameraOrigin', 'CameraDirection']:
        if col in df.columns:
            coords = ['X', 'Y', 'Z']
            try:
                 # Handle cases where rows might have None or NaN before stacking
                valid_entries = df[col].dropna()
                if not valid_entries.empty:
                    split_df = pd.DataFrame(valid_entries.tolist(), index=valid_entries.index, columns=[f'{col}_{c}' for c in coords])
                    df = df.join(split_df)
                else:
                     for c in coords: # Create NaN columns if no valid data
                         df[f'{col}_{c}'] = np.nan
            except Exception as e:
                 print(f"Error expanding column {col} in {file_path}: {e}")
                 for c in coords: # Create NaN columns on error
                     df[f'{col}_{c}'] = np.nan
            df = df.drop(columns=[col]) # Drop original list column

    df = df.sort_values(by='Timestamp').reset_index(drop=True)
    # Calculate time difference in seconds
    df['Time_sec'] = (df['Timestamp'] - df['Timestamp'].iloc[0]) / 1e7 # Assuming timestamp is in 100ns increments
    df['dt'] = df['Time_sec'].diff().fillna(0.0)

    return df


def load_all_data(data_folder="test_1"):
    """Loads all XML files from the specified folder."""
    xml_files = glob.glob(os.path.join(data_folder, "*.xml"))
    all_data = {}
    print(f"Found {len(xml_files)} XML files in {data_folder}")
    for file_path in tqdm(xml_files, desc="Loading XML files"):
        file_name = os.path.basename(file_path)
        df = parse_xml_file(file_path)
        if df is not None and not df.empty:
            # Extract user ID and task type from filename (adjust pattern if needed)
            parts = file_name.replace('.xml', '').split('_')
            user_id = parts[0] # Assuming user ID is the first part
            task = parts[-1]   # Assuming task is the last part ('truss' or 'warehouse')
            key = f"{user_id}_{task}"
            all_data[key] = df
            print(f"Loaded {file_name} ({len(df)} rows) as key '{key}'")
        else:
            print(f"Skipped {file_name} due to parsing errors or empty data.")

    # Separate truss and warehouse data
    truss_data = {k: v for k, v in all_data.items() if k.endswith('_truss')}
    warehouse_data = {k: v for k, v in all_data.items() if k.endswith('_warehouse')}

    print(f"\nLoaded {len(truss_data)} truss datasets.")
    print(f"Loaded {len(warehouse_data)} warehouse datasets.")

    return truss_data, warehouse_data

def create_windows(df, window_size_points):
    """Breaks a DataFrame into overlapping windows."""
    windows = []
    # Simple non-overlapping windows for now. Overlap can be added later if needed.
    for i in range(0, len(df) - window_size_points + 1, window_size_points):
         windows.append(df.iloc[i : i + window_size_points])
    return windows

def anonymize_window(window_df):
    """
    Anonymizes a single window by making positions and orientations relative.
    Focuses on velocity, acceleration, and relative changes rather than absolute coords.
    """
    if window_df.empty or len(window_df) < 2:
        return window_df # Cannot process empty or single-row windows

    anonymized = window_df.copy()

    # 1. Anonymize Camera Position/Direction: Use velocity and turning rate
    # Calculate delta position and time
    delta_pos = anonymized[['CameraOrigin_X', 'CameraOrigin_Y', 'CameraOrigin_Z']].diff()
    delta_time = anonymized['dt']

    # Calculate velocity (handling potential division by zero)
    velocity = delta_pos.div(delta_time, axis=0)
    anonymized['CameraVelocity_X'] = velocity['CameraOrigin_X']
    anonymized['CameraVelocity_Y'] = velocity['CameraOrigin_Y']
    anonymized['CameraVelocity_Z'] = velocity['CameraOrigin_Z']
    anonymized['CameraSpeed'] = np.linalg.norm(velocity.fillna(0).values, axis=1)

    # Calculate turning rate (angle change between consecutive direction vectors)
    cam_dir_cols = ['CameraDirection_X', 'CameraDirection_Y', 'CameraDirection_Z']
    directions = anonymized[cam_dir_cols].values
    # Normalize directions to avoid issues with magnitude
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    valid_norms_mask = (norms > 1e-6).flatten() # Avoid division by zero/small numbers
    normalized_directions = np.full_like(directions, np.nan) # Initialize with NaNs
    if np.any(valid_norms_mask):
         normalized_directions[valid_norms_mask] = directions[valid_norms_mask] / norms[valid_norms_mask]


    # Calculate dot product between consecutive normalized directions
    dot_products = np.einsum('ij,ij->i', normalized_directions[:-1], normalized_directions[1:])
    # Clamp dot products to [-1, 1] to avoid domain errors in arccos
    dot_products = np.clip(dot_products, -1.0, 1.0)
    angles = np.arccos(dot_products) # Angle in radians
    # Convert angles to degrees and calculate angular velocity (deg/s)
    angular_velocity = np.degrees(angles) / delta_time[1:].values # Use delta_time from the second row onwards
    # Prepend NaN or 0 for the first element
    anonymized['CameraTurningRate'] = np.concatenate(([np.nan], angular_velocity))


    # 2. Anonymize Gaze Position/Direction: Use relative gaze coords or gaze velocity
    # Similar calculations for Gaze Origin velocity/speed if needed
    gaze_origin_cols = ['GazeOrigin_X', 'GazeOrigin_Y', 'GazeOrigin_Z']
    if all(col in anonymized.columns for col in gaze_origin_cols):
         delta_gaze_pos = anonymized[gaze_origin_cols].diff()
         gaze_velocity = delta_gaze_pos.div(delta_time, axis=0)
         anonymized['GazeVelocity_X'] = gaze_velocity['GazeOrigin_X']
         anonymized['GazeVelocity_Y'] = gaze_velocity['GazeOrigin_Y']
         anonymized['GazeVelocity_Z'] = gaze_velocity['GazeOrigin_Z']
         anonymized['GazeSpeed'] = np.linalg.norm(gaze_velocity.fillna(0).values, axis=1)


    # Gaze direction turning rate (similar to camera)
    gaze_dir_cols = ['GazeDirection_X', 'GazeDirection_Y', 'GazeDirection_Z']
    if all(col in anonymized.columns for col in gaze_dir_cols):
        gaze_directions = anonymized[gaze_dir_cols].values
        gaze_norms = np.linalg.norm(gaze_directions, axis=1, keepdims=True)
        gaze_valid_norms_mask = (gaze_norms > 1e-6).flatten()
        normalized_gaze_directions = np.full_like(gaze_directions, np.nan)
        if np.any(gaze_valid_norms_mask):
            normalized_gaze_directions[gaze_valid_norms_mask] = gaze_directions[gaze_valid_norms_mask] / gaze_norms[gaze_valid_norms_mask]

        gaze_dot_products = np.einsum('ij,ij->i', normalized_gaze_directions[:-1], normalized_gaze_directions[1:])
        gaze_dot_products = np.clip(gaze_dot_products, -1.0, 1.0)
        gaze_angles = np.arccos(gaze_dot_products)
        gaze_angular_velocity = np.degrees(gaze_angles) / delta_time[1:].values
        anonymized['GazeTurningRate'] = np.concatenate(([np.nan], gaze_angular_velocity))


    # Drop original absolute position/direction columns? Decide based on feature needs.
    # For now, keep them, but features should primarily use derived velocity/rate values.
    # Example: Drop columns (optional)
    # anonymized = anonymized.drop(columns=[
    #     'CameraOrigin_X', 'CameraOrigin_Y', 'CameraOrigin_Z',
    #     'CameraDirection_X', 'CameraDirection_Y', 'CameraDirection_Z',
    #     'GazeOrigin_X', 'GazeOrigin_Y', 'GazeOrigin_Z',
    #     'GazeDirection_X', 'GazeDirection_Y', 'GazeDirection_Z',
    #     'IntersectionPoint_X', 'IntersectionPoint_Y', 'IntersectionPoint_Z',
    #     'ObjectPosition_X', 'ObjectPosition_Y', 'ObjectPosition_Z',
    #     'PositionOnDisplayX', 'PositionOnDisplayY' # These are relative to display already, maybe keep?
    # ])


    # Fill NaNs created by diff() or calculations (e.g., first row) - use forward fill or fill with 0
    anonymized = anonymized.fillna(0) # Or use ffill() / bfill() if appropriate

    return anonymized
