# main_analysis.ipynb
def create_windows(df, window_size_points, stride):
    """Breaks a DataFrame into overlapping windows."""
    windows = []
    for i in range(0, len(df) - window_size_points + 1, stride):
        windows.append(df.iloc[i : i + window_size_points])
    return windows

# data_loader.py
STRIDE = 10  # 1 = max overlap, 60 = no overlap (with 60Hz and 1s window)
windows_raw = data_loader.create_windows(df, WINDOW_POINTS, STRIDE)

# anonymized window
if all(col in anonymized.columns for col in ['GazeOrigin_X', 'CameraOrigin_X']):
    anonymized['RelGaze_X'] = anonymized['GazeOrigin_X'] - anonymized['CameraOrigin_X']
    anonymized['RelGaze_Y'] = anonymized['GazeOrigin_Y'] - anonymized['CameraOrigin_Y']
    anonymized['RelGaze_Z'] = anonymized['GazeOrigin_Z'] - anonymized['CameraOrigin_Z']
