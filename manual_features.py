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
        # Return default values or NaNs if window is too short
        # Define default/NaN structure based on expected feature names    
        return {
            # Eye metrics
            'avg_pupil_diameter': np.nan,
            #'avg_pupil_velocity': np.nan,
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
            
            # Movement
            'avg_movement_velocity_mps': np.nan,
            'avg_turning_velocity_degps': np.nan,
            'total_action': np.nan
        }

    dt = 1.0/sampling_rate  # seconds per sample

    # Eye dynamics
    eye_dir = window_df[['GazeDirection_X', 'GazeDirection_Y', 'GazeDirection_Z']]
    pupil_diam = window_df['PupilDiameter']

    # Movement
    movement = window_df[['CameraOrigin_X', 'CameraOrigin_Y', 'CameraOrigin_Z']]
    turning = window_df[['CameraDirection_X', 'CameraDirection_Y', 'CameraDirection_Z']]
    
    window_df['movement_velocity'] = np.linalg.norm(movement.diff(), axis=1) / dt
    window_df['turning_velocity'] = np.linalg.norm(turning.diff(), axis=1) / dt
    
    window_df['eye_velocity_lin'] = np.linalg.norm(eye_dir.diff(), axis=1) / dt
    window_df['eye_accel_lin'] = window_df['eye_velocity_lin'].diff() / dt

    window_df['pupil_velocity'] = pupil_diam.diff().abs() / dt
    

    # Compute relative gaze (egocentric coordinates)
    if all(col in window_df.columns for col in ['GazeOrigin_X', 'CameraOrigin_X']):
        rel_gaze = window_df[['GazeOrigin_X', 'GazeOrigin_Y', 'GazeOrigin_Z']].values - \
                    window_df[['CameraOrigin_X', 'CameraOrigin_Y', 'CameraOrigin_Z']].values
        rel_gaze_mag = np.linalg.norm(rel_gaze, axis=1)
        # Average distance of gaze point from head/camera
        features['rel_gaze_distance'] = np.nanmean(rel_gaze_mag)
        # Gaze stability: how much the relative gaze position changes
        rel_gaze_stability = np.diff(rel_gaze, axis=0)
        stability = np.linalg.norm(rel_gaze_stability, axis=1).mean() if len(rel_gaze_stability) > 0 else np.nan
        features['gaze_stability'] = stability
    else:
        features['rel_gaze_distance'] = np.nan
        features['gaze_stability'] = np.nan


    '''#delete rows with NaN values
    window_df = window_df.dropna()
    '''

    # Gaze area covered
    gaze_points = window_df[['GazeOrigin_X', 'GazeOrigin_Y']]
    spatial_var = gaze_points.var()
    features['gaze_area_covered'] = spatial_var.sum()  # X_var + Y_var

    # Fixation and saccade estimation (based on velocity threshold)
    velocity_threshold = 30  # deg/s
    normed_eye_dir = eye_dir.div(np.linalg.norm(eye_dir, axis=1), axis=0)
    # Compute cosine between direction vectors (successive rows)
    cos_theta = (normed_eye_dir.shift(1) * normed_eye_dir).sum(axis=1).clip(-1.0, 1.0)
    theta_rad = np.arccos(cos_theta)
    # Fix first angle using forward diff
    theta_rad.iloc[0] = np.arccos(np.clip((normed_eye_dir.iloc[0] * normed_eye_dir.iloc[1]).sum(), -1.0, 1.0))

    # Calculate angular velocity and acceleration
    window_df['eye_velocity_deg'] = np.degrees(theta_rad) / dt  # Convert to deg/sec
    window_df['eye_accel_deg'] = window_df['eye_velocity_deg'].diff() / dt


    is_fixation = window_df['eye_velocity_deg'] < velocity_threshold
    n_fix = is_fixation.sum()
    features['n_fixations'] = int(n_fix)

    if n_fix > 0:
        features['avg_fixation_duration'] = n_fix * dt
        fixation_gaze = gaze_points[is_fixation]
        if len(fixation_gaze) > 1:
            features['fixation_spatial_variance'] = fixation_gaze.var().mean()
        else:
            features['fixation_spatial_variance'] = np.nan
        features['fix_sacc_ratio'] = n_fix / len(window_df)
    else:
        features['avg_fixation_duration'] = np.nan  # or np.nan
        features['fixation_spatial_variance'] = np.nan  # sentinel value
        features['fix_sacc_ratio'] = np.nan


    # Saccades: where eye_velocity_deg exceeds threshold
    saccades = window_df['eye_velocity_deg'] >= velocity_threshold

    # Mean saccade velocity
    if saccades.any():
        features['mean_saccade_velocity'] = window_df['eye_velocity_deg'][saccades].mean()
    else:
        features['mean_saccade_velocity'] = np.nan

    # Calculate norm of directional changes
    gaze_diff = eye_dir.diff().apply(np.linalg.norm, axis=1)

    # Drop the first row (NaN) so indices align with saccades
    gaze_diff.iloc[0] = np.nan  # Ensure safe NaN at index 0
    saccade_amplitudes = gaze_diff[saccades]

    # Compute mean if valid
    if saccade_amplitudes.dropna().any():
        features['mean_saccade_amplitude'] = saccade_amplitudes.mean()
    else:
        features['mean_saccade_amplitude'] = np.nan



    # Basic averages
    features['avg_pupil_diameter'] = window_df['PupilDiameter'].mean()
    #features['avg_pupil_velocity'] = window_df['pupil_velocity'].mean()
    features['avg_eye_velocity_degps'] = window_df['eye_velocity_deg'].mean()
    features['avg_eye_accel_degps2'] = window_df['eye_accel_deg'].mean()
    features['avg_eye_velocity_lin'] = window_df['eye_velocity_lin'].mean()
    features['avg_eye_accel_lin'] = window_df['eye_accel_lin'].mean()

    features['avg_movement_velocity_mps'] = window_df['movement_velocity'].mean()
    features['avg_turning_velocity_degps'] = window_df['turning_velocity'].mean()
    features['total_action'] = (window_df['movement_velocity'] + window_df['turning_velocity']).sum()

    #window_df.drop(columns=['eye_velocity_deg', 'eye_accel_deg', 'movement_velocity', 'turning_velocity'], inplace=True, errors='ignore')


    return features