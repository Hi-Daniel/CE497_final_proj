"""
Manual feature extraction pipeline:

raw data -> identify gaps -> fill gaps -> noise filter -> fixation classification -> windowing -> calculate features
"""


import pandas as pd
import numpy as np

def calculate_manual_features_for_window(window_df, sampling_rate=60):
    """
    Calculates manual features for a single anonymized window (DataFrame).
    
    Returns a dictionary of features:
        Eye behavior:
        - Average pupil diameter (mm)
        - Average eye movement velocity (deg/s)
        - Average eye movement acceleration (deg/s²)
        - Ratio of fixation/saccade
        - Number of fixations
        - Average fixation duration (s)
        - Spatial variance of fixations (deg²)
        - Mean saccade velocity (deg/s)
        - Mean saccade amplitude (deg)
        - Relative gaze distance (mm)
        - Gaze stability (mean change in RelGaze vector magnitude)

        Movement:
        - Average movement velocity (m/s)
        - Average turning velocity (deg/s)
        - Total action (translation + rotation)

        Spatial distribution:
        - Gaze area covered (deg²)
        - Gaze spatial density (samples / deg²)
    """
        
    features = {}

    if window_df.empty or len(window_df) < 2:
        return {
            'avg_pupil_diameter': np.nan,
            'avg_eye_velocity_deg_degps': np.nan,
            'avg_eye_accel_deg_degps2': np.nan,
            'fix_sacc_ratio': np.nan,
            'n_fixations': 0,
            'avg_fixation_duration': np.nan,
            'fixation_spatial_variance': np.nan,
            'mean_saccade_velocity': np.nan,
            'mean_saccade_amplitude': np.nan,
            'gaze_area_covered': np.nan,
            'gaze_spatial_density': np.nan,
            'rel_gaze_distance': np.nan,
            'gaze_stability': np.nan,
            'avg_movement_velocity_mps': np.nan,
            'avg_turning_velocity_degps': np.nan,
            'total_action': np.nan
        }

    dt = 1.0 / sampling_rate
    features = {}

    # Core computations
    compute_kinematics_if_missing(window_df, dt)
    features.update(compute_eye_movement_features(window_df, dt))
    features.update(compute_fixation_saccade_features(window_df, dt))
    features.update(compute_gaze_area_features(window_df))
    features.update(compute_relative_gaze_features(window_df))
    features.update(compute_movement_features(window_df))

    return features

def compute_kinematics_if_missing(df, dt):
    eye_dir = df[['GazeDirection_X', 'GazeDirection_Y', 'GazeDirection_Z']]
    if 'eye_velocity_lin' not in df.columns:
        df['eye_velocity_lin'] = np.linalg.norm(eye_dir.diff(), axis=1) / dt
    if 'eye_accel_lin' not in df.columns:
        df['eye_accel_lin'] = df['eye_velocity_lin'].diff() / dt

    if 'pupil_velocity' not in df.columns and 'PupilDiameter' in df.columns:
        df['pupil_velocity'] = df['PupilDiameter'].diff().abs() / dt

    if 'movement_velocity' not in df.columns:
        move = df[['CameraOrigin_X', 'CameraOrigin_Y', 'CameraOrigin_Z']]
        df['movement_velocity'] = np.linalg.norm(move.diff(), axis=1) / dt

    if 'turning_velocity' not in df.columns:
        turn = df[['CameraDirection_X', 'CameraDirection_Y', 'CameraDirection_Z']]
        df['turning_velocity'] = np.linalg.norm(turn.diff(), axis=1) / dt

    if 'eye_velocity_deg' not in df.columns:
        normed_eye = eye_dir.div(np.linalg.norm(eye_dir, axis=1), axis=0)
        cos_theta = (normed_eye.shift(1) * normed_eye).sum(axis=1).clip(-1.0, 1.0)
        theta_rad = np.arccos(cos_theta)
        theta_rad.iloc[0] = np.arccos(np.clip((normed_eye.iloc[0] * normed_eye.iloc[1]).sum(), -1.0, 1.0))
        df['eye_velocity_deg'] = np.degrees(theta_rad) / dt
    if 'eye_accel_deg' not in df.columns:
        df['eye_accel_deg'] = df['eye_velocity_deg'].diff() / dt

def compute_eye_movement_features(df, dt):
    return {
        'avg_pupil_diameter': df['PupilDiameter'].mean(),
        'avg_eye_velocity_degps': df['eye_velocity_deg'].mean(),
        'avg_eye_accel_degps2': df['eye_accel_deg'].mean(),
        'avg_eye_velocity_lin': df['eye_velocity_lin'].mean(),
        'avg_eye_accel_lin': df['eye_accel_lin'].mean()
    }

def compute_fixation_saccade_features(df, dt, velocity_threshold=30):
    features = {}
    gaze_points = df[['GazeOrigin_X', 'GazeOrigin_Y']]
    eye_dir = df[['GazeDirection_X', 'GazeDirection_Y', 'GazeDirection_Z']]
    
    is_fixation = df['eye_velocity_deg'] < velocity_threshold
    n_fix = is_fixation.sum()
    features['n_fixations'] = int(n_fix)

    if n_fix > 0:
        features['avg_fixation_duration'] = n_fix * dt
        features['fixation_spatial_variance'] = gaze_points[is_fixation].var().mean()
        features['fix_sacc_ratio'] = n_fix / len(df)
    else:
        features['avg_fixation_duration'] = np.nan
        features['fixation_spatial_variance'] = np.nan
        features['fix_sacc_ratio'] = np.nan

    saccades = df['eye_velocity_deg'] >= velocity_threshold
    features['mean_saccade_velocity'] = df['eye_velocity_deg'][saccades].mean() if saccades.any() else np.nan

    gaze_diff = eye_dir.diff().apply(np.linalg.norm, axis=1)
    gaze_diff.iloc[0] = np.nan
    features['mean_saccade_amplitude'] = gaze_diff[saccades].mean() if gaze_diff[saccades].dropna().any() else np.nan

    return features

def compute_gaze_area_features(df):
    features = {}
    if all(col in df.columns for col in ['X_2d_X', 'X_2d_Y']):
        spatial_var = df[['X_2d_X', 'X_2d_Y']].var()
        features['gaze_area_covered'] = spatial_var.sum()
    else:
        features['gaze_area_covered'] = np.nan

    head_var = df[['GazeOrigin_X', 'GazeOrigin_Y']].var()
    features['head_area_covered'] = head_var.sum()
    return features

def make_nan_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Look at the PupilDiameter column and if row is Nan, make eye related columns also Nan.
    """
    gaze_cols = ['GazeOrigin_X', 'GazeOrigin_Y', 'GazeOrigin_Z', 'GazeDirection_X', 'GazeDirection_Y', 'GazeDirection_Z']
    df.loc[df['PupilDiameter'].isna(), gaze_cols] = np.nan
    return df

def index_by_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Index the dataframe by the Time_sec column. and convert to timedelta
    """
    # make Time_sec column a datetime column and index the dataframe by it
    df['Time_timedelta'] = pd.to_timedelta(df['Time_sec'], unit='s')
    df = df.set_index('Time_timedelta')
    return df

def identify_gaps(df: pd.DataFrame, max_gap_duration_ms=75) -> pd.DataFrame:
    """
    Identify gaps in data make temporary column "gap_fill" to indicate if the gap should be filled or not.
    a gap should be fille if it is less than max_gap_duration_ms.
    """
    # make temporary column "gap_fill" to indicate if the gap should be filled or not.
    df['gap_fill'] = False
    # iterate through rows in df
    start_time = None
    end_time = None
    for index, row in df.iterrows():
        if np.isnan(row['PupilDiameter']) and start_time is None:
            start_time = index  
        elif not np.isnan(row['PupilDiameter']) and start_time is not None:
            end_time = index
            if end_time - start_time < pd.Timedelta(max_gap_duration_ms, unit='ms'):
                df.loc[start_time:end_time, 'gap_fill'] = True
            start_time = None
            end_time = None
    return df

def fill_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill in gaps in dataframe via interpolation if gap_fill is True.
    """
    # Make a copy and interpolate all values
    df_filled = df.copy()
    df_filled = df_filled.interpolate(method='linear', limit_direction='both', limit_area='inside')
    
    # replace values in original df where gap_fill is True with interpolated values
    df.loc[df['gap_fill'], :] = df_filled.loc[df['gap_fill'], :]
    
    return df

def noise_filter(df: pd.DataFrame, width_ms=10) -> pd.DataFrame:
    """
    Apply low pass filter to gaze related columns.
    """
    # make a copy of the dataframe
    gaze_cols = ['X_2d_X', 'X_2d_Y', 'GazeOrigin_X', 'GazeOrigin_Y', 'GazeOrigin_Z', 'GazeDirection_X', 'GazeDirection_Y', 'GazeDirection_Z']
    df_filtered = df.copy()
    # apply low pass filter to gaze related columns
    df_filtered[gaze_cols] = df_filtered[gaze_cols].apply(lambda x: x.rolling(window=f"{width_ms}ms", min_periods=1).mean())
    return df_filtered

def calculate_velocity_acceleration_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate velocity and acceleration columns from gaze related columns.
    """
    # make a copy of the dataframe
    df_filtered = df.copy()
    # Get dt for each row by calculating the difference between consecutive indices
    dt_series = df_filtered.index.to_series().diff()
    # calculate velocity columns
    # Turn direction vectors into spherical angles
    df_filtered["GazeDirection_Theta"] = np.arctan2(df_filtered["GazeDirection_Y"], df_filtered["GazeDirection_X"]) * 180 / np.pi
    df_filtered["GazeDirection_Phi"] = np.arccos(df_filtered["GazeDirection_Y"] / 
                                                np.linalg.norm(df_filtered[["GazeDirection_X", "GazeDirection_Y", "GazeDirection_Z"]], axis=1)) * 180/np.pi

    # Calculate head movement as the difference between camera position and gaze origin
    df_filtered["HeadLocation_X"] = df_filtered["GazeOrigin_X"] - df_filtered["CameraOrigin_X"]
    df_filtered["HeadLocation_Y"] = df_filtered["GazeOrigin_Y"] - df_filtered["CameraOrigin_Y"] 
    df_filtered["HeadLocation_Z"] = df_filtered["GazeOrigin_Z"] - df_filtered["CameraOrigin_Z"]

    # Calculate velocities and accelerations
    dt_series = df_filtered.index.to_series().diff().dt.total_seconds()

    df_filtered["Velocity_2d"] = np.linalg.norm(df_filtered[["X_2d_X", "X_2d_Y"]].diff(), axis=1) / dt_series
    df_filtered["CameraMovementVelocity_m_s"] = np.linalg.norm(df_filtered[["CameraOrigin_X", "CameraOrigin_Y", "CameraOrigin_Z"]].diff(), axis=1) / dt_series
    df_filtered["CameraMovementAcceleration_m_s"] = df_filtered["CameraMovementVelocity_m_s"].diff() / dt_series
    df_filtered["HeadMovementVelocity_m_s"] = np.linalg.norm(df_filtered[["HeadLocation_X", "HeadLocation_Y", "HeadLocation_Z"]].diff(), axis=1) / dt_series
    df_filtered["HeadMovementAcceleration_m_s_s"] = df_filtered["HeadMovementVelocity_m_s"].diff() / dt_series
    
    # Calculate turning velocity and acceleration
    # Calculate angular differences accounting for wraparound at 180/-180 degrees
    theta_diff = df_filtered["GazeDirection_Theta"].diff()
    theta_diff = np.where(theta_diff > 180, theta_diff - 360, theta_diff)
    theta_diff = np.where(theta_diff < -180, theta_diff + 360, theta_diff)
    
    phi_diff = df_filtered["GazeDirection_Phi"].diff()
    phi_diff = np.where(phi_diff > 180, phi_diff - 360, phi_diff)
    phi_diff = np.where(phi_diff < -180, phi_diff + 360, phi_diff)
    
    # Calculate turning velocity using the corrected angular differences
    df_filtered["TurningVelocity_deg_s"] = np.sqrt(theta_diff**2 + phi_diff**2) / dt_series
    df_filtered["TurningAcceleration_deg_s"] = df_filtered["TurningVelocity_deg_s"].diff() / dt_series
    return df_filtered

def fixation_classification(df: pd.DataFrame, velocity_threshold_deg_s=30) -> pd.DataFrame:
    """
    Classify fixations using 3D I-VT.
    Creates new column "is_fixation" with values True or False.
    """
    # make a copy of the dataframe
    df_filtered = df.copy()

    # Calculate eye theta as the angle difference between gaze direction and previous gaze direction
    eye_dir = df_filtered[['GazeDirection_X', 'GazeDirection_Y', 'GazeDirection_Z']].to_numpy()
    eye_dir_prev = np.roll(eye_dir, 1, axis=0)
    eye_dir_prev[np.isnan(eye_dir_prev)] = 0
    df_filtered["eye_theta"] = np.arccos(np.sum(eye_dir * eye_dir_prev, axis=1) 
                                         / np.linalg.norm(eye_dir, axis=1) 
                                         / np.linalg.norm(eye_dir_prev, axis=1))
    df_filtered["eye_theta"] = df_filtered["eye_theta"] * 180 / np.pi

    # Apply 5 tap filter to eye_theta with weighted average 
    weights = np.array([1, 2, 3, 2, 1])
    dt_series = df_filtered.index.to_series().diff().dt.total_seconds()
    df_filtered["eye_theta"] = df_filtered["eye_theta"].rolling(window=5, min_periods=5).apply(lambda x: np.sum(x * weights) / np.sum(weights))
    df_filtered["eye_velocity_deg_s"] = df_filtered["eye_theta"] / dt_series.rolling(window=5, min_periods=5).sum()
 

    # Apply threshold to classify fixations
    df_filtered["is_fixation"] = df_filtered["eye_velocity_deg_s"] < velocity_threshold_deg_s
    return df_filtered