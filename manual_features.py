import pandas as pd
import numpy as np

def calculate_manual_features_for_window(window_df):
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
            'avg_movement_velocity_mps': np.nan, 'avg_turning_velocity_degps': np.nan,
            'total_action': np.nan
        }


    # calculate movement velocity and turning velocity
    window_df['movement_velocity'] = np.linalg.norm(window_df[['CameraOrigin_X', 'CameraOrigin_Y', 'CameraOrigin_Z']].diff(), axis=1)
    window_df['turning_velocity'] = np.linalg.norm(window_df[['CameraDirection_X', 'CameraDirection_Y', 'CameraDirection_Z']].diff(), axis=1)

    # calculate eye turning velocity and acceleration
    window_df['eye_velocity'] = np.linalg.norm(window_df[['GazeDirection_X', 'GazeDirection_Y', 'GazeDirection_Z']].diff(), axis=1)
    window_df['eye_accel'] = np.linalg.norm(window_df[['GazeDirection_X', 'GazeDirection_Y', 'GazeDirection_Z']].diff().diff(), axis=1)

    #delete rows with NaN values
    window_df = window_df.dropna()
    # calculate eye metrics
    features['avg_pupil_diameter'] = window_df['pupil'].mean()
    features['avg_eye_velocity_degps'] = window_df['eye_velocity'].mean()
    features['avg_eye_accel_degps2'] = window_df['eye_accel'].mean()


    return features
