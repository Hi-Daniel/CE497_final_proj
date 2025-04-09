import pandas as pd
import numpy as np

def calculate_manual_features_for_window(window_df, sampling_rate=60):
    """
    Calculates the 14 manual features for a single anonymized window (DataFrame).
    
    returns a dictionary of features:
        - Average pupil diameter (mm)
        - Average eye movement velocity (deg/s)
        - Average eye movement acceleration (deg/s/s)
        - Ratio of fixation/saccade (s)
        - Number of fixations
        - Average fixation duration (s)
        - Spatial variance of fixations (deg)
        - Mean saccade velocity (deg/s)
        - Mean saccade amplitude (deg)
        - Average movement velocity (m/s)
        - Average turning velocity (deg/s)
        - Total action, including translation and rotation
    """
    features = {}

    if window_df.empty or len(window_df) < 2:
        # Return default values or NaNs if window is too short
        # Define default/NaN structure based on expected feature names    
        return {
            # Eye metrics
            'avg_pupil_diameter': np.nan, 'avg_eye_velocity_degps': np.nan,
            'avg_eye_accel_degps2': np.nan, 'fix_sacc_ratio': np.nan,
            'n_fixations': 0, 'avg_fixation_duration': np.nan,
            'fixation_spatial_variance': np.nan, 'mean_saccade_velocity': np.nan,
            'mean_saccade_amplitude': np.nan, 'gaze_area_covered': np.nan,
            'gaze_spatial_density': np.nan, 
            # Movement patterns
            'avg_movement_velocity_mps': np.nan,
            'avg_turning_velocity_degps': np.nan, 'total_action': np.nan
            }

    custom_fill = {
        'PupilDiameter': 'ffill',
        #'X_2d_X': 'interpolate',
        #'X_2d_Y': 'interpolate',
        #'SomeSensorColumn': 'mean'
    }

    window_df = clean_window_df(window_df, custom_fill=custom_fill, normalize_dirs=True)
    dt = 1.0/sampling_rate  # seconds per sample

    # Movement
    movement = window_df[['CameraOrigin_X', 'CameraOrigin_Y', 'CameraOrigin_Z']]
    turning = window_df[['CameraDirection_X', 'CameraDirection_Y', 'CameraDirection_Z']]
    window_df['movement_velocity'] = np.linalg.norm(movement.diff(), axis=1) / dt
    window_df['turning_velocity'] = np.linalg.norm(turning.diff(), axis=1) / dt

    # Eye dynamics
    eye_dir = window_df[['GazeDirection_X', 'GazeDirection_Y', 'GazeDirection_Z']]
    window_df['eye_velocity'] = np.linalg.norm(eye_dir.diff(), axis=1) / dt
    window_df['eye_accel'] = window_df['eye_velocity'].diff() / dt

    '''#delete rows with NaN values
    window_df = window_df.dropna()
    '''
    
    # Gaze area covered
    gaze_points = window_df[['GazeOrigin_X', 'GazeOrigin_Y']]
    area_x = gaze_points['GazeOrigin_X'].max() - gaze_points['GazeOrigin_X'].min()
    area_y = gaze_points['GazeOrigin_Y'].max() - gaze_points['GazeOrigin_Y'].min()
    features['gaze_area_covered'] = area_x * area_y
    features['gaze_spatial_density'] = len(gaze_points) / (features['gaze_area_covered'] + 1e-5)

    # Fixation and saccade estimation (based on velocity threshold)
    velocity_threshold = 20  # deg/s, rough estimate
    is_fixation = window_df['eye_velocity'] < velocity_threshold
    n_fix = is_fixation.sum()
    features['n_fixations'] = int(n_fix)

    if n_fix > 0:
        features['avg_fixation_duration'] = n_fix * dt
        features['fixation_spatial_variance'] = gaze_points[is_fixation].var().mean()
        features['fix_sacc_ratio'] = n_fix / len(window_df)
    else:
        features['avg_fixation_duration'] = np.nan
        features['fixation_spatial_variance'] = np.nan
        features['fix_sacc_ratio'] = 0

    # Saccades: where eye_velocity exceeds threshold
    saccades = window_df['eye_velocity'] >= velocity_threshold
    if saccades.any():
        features['mean_saccade_velocity'] = window_df['eye_velocity'][saccades].mean()
        amplitudes = eye_dir.diff().loc[saccades].apply(np.linalg.norm, axis=1)
        features['mean_saccade_amplitude'] = amplitudes.mean()
    else:
        features['mean_saccade_velocity'] = np.nan
        features['mean_saccade_amplitude'] = np.nan

    # Basic averages
    features['avg_pupil_diameter'] = window_df['PupilDiameter'].mean()
    features['avg_eye_velocity_degps'] = window_df['eye_velocity'].mean()
    features['avg_eye_accel_degps2'] = window_df['eye_accel'].mean()

    features['avg_movement_velocity_mps'] = window_df['movement_velocity'].mean()
    features['avg_turning_velocity_degps'] = window_df['turning_velocity'].mean()
    features['total_action'] = (window_df['movement_velocity'] + window_df['turning_velocity']).sum()

    #window_df.drop(columns=['eye_velocity', 'eye_accel', 'movement_velocity', 'turning_velocity'], inplace=True, errors='ignore')


    return features


def clean_window_df(window_df, fill_strategy='interpolate', custom_fill=None, normalize_dirs=True, verbose=False):
    """
    Cleans missing data in a window DataFrame.

    Args:
        window_df (pd.DataFrame): The input window.
        fill_strategy (str): Default strategy for filling all columns ('interpolate', 'ffill', 'zero', 'drop').
        custom_fill (dict): Optional dict to apply specific strategies to specific columns.
        normalize_dirs (bool): Whether to normalize direction vectors (gaze/camera).
        verbose (bool): If True, prints info about NaNs and normalization.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    df = window_df.copy()
    if df.empty:
        return df

    if verbose:
        nan_cols = df.columns[df.isnull().any()].tolist()
        if nan_cols:
            print(f"NaNs before cleaning in: {nan_cols}")

    # --- 1. Apply custom fill logic first (overrides default strategy) ---
    if custom_fill:
        for col, method in custom_fill.items():
            if col in df.columns:
                if method == 'ffill':
                    df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
                elif method == 'mean':
                    df[col] = df[col].fillna(df[col].mean())
                elif method == 'zero':
                    df[col] = df[col].fillna(0)
                elif method == 'interpolate':
                    df[col] = df[col].interpolate(method='linear', limit_direction='both')
                elif method == 'drop':
                    df = df.dropna(subset=[col])
                else:
                    raise ValueError(f"Unknown fill method '{method}' for column '{col}'")

    # --- 2. Apply global fill strategy to remaining NaNs ---
    if fill_strategy and fill_strategy not in ['none']:
        if fill_strategy == 'interpolate':
            df = df.interpolate(method='linear', limit_direction='both')
        elif fill_strategy == 'ffill':
            df = df.fillna(method='ffill').fillna(method='bfill')
        elif fill_strategy == 'zero':
            df = df.fillna(0)
        elif fill_strategy == 'drop':
            df = df.dropna()
        else:
            raise ValueError(f"Unknown global fill_strategy: '{fill_strategy}'")

    # --- 3. Normalize direction vectors ---
    if normalize_dirs:
        direction_groups = [
            ['GazeDirection_X', 'GazeDirection_Y', 'GazeDirection_Z'],
            ['CameraDirection_X', 'CameraDirection_Y', 'CameraDirection_Z']
        ]
        for dir_cols in direction_groups:
            if all(col in df.columns for col in dir_cols):
                norm = np.linalg.norm(df[dir_cols], axis=1)
                norm[norm == 0] = 1  # avoid divide-by-zero
                df[dir_cols] = df[dir_cols].div(norm, axis=0)

    return df


def normalize_vector_cols(df, cols):
    norm = np.linalg.norm(df[cols], axis=1)
    df[cols] = df[cols].div(norm, axis=0)
    return df