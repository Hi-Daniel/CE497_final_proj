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