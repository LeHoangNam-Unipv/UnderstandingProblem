import os
import pandas as pd
import numpy as np
from scipy.stats import gmean, skew, iqr, kurtosis, median_abs_deviation

#NUMBER_OF_MIN_ROW = 9300
NUMBER_OF_MIN_ROW = 16800
SEQUENCE_LENGTH = 1500
FOLDER = r"C:\Users\lehoa\Downloads\data-20240513T042351Z-001\data\P4"

NUMBER_OF_FEATURES = 70
# Function to calculate statistics
def calculate_statistics(data, include_gmean=True):
    stats = {
        'min': np.min(data),
        'max': np.max(data),
        'mean': np.mean(data),
        'median': np.median(data),
        'std': np.std(data),
        'median_abs_deviation': median_abs_deviation(data),
        'skew': skew(data),
        'iqr': iqr(data),
        'kurtosis': kurtosis(data)
    }
    if include_gmean:
        stats['gmean'] = gmean(data[data > 0]) if np.any(data > 0) else 0
    return stats

# Function to process a chunk of 300 rows
def process_chunk(chunk):
    # 1. Number of Fixations (unique FPOGID)
    num_fixations = chunk['FPOGID'].nunique()

    # 2-10. Statistics of fixation duration (FPOGD for each fixation)
    fixation_durations = chunk.groupby('FPOGID')['FPOGD'].last()
    duration_stats = calculate_statistics(fixation_durations)

    # 11-19. Statistics of saccade distance
    fixation_coords = chunk.groupby('FPOGID')[['FPOGX', 'FPOGY']].mean()
    if len(fixation_coords) > 1:
        fixation_distances = np.sqrt(np.sum(np.diff(fixation_coords, axis=0)**2, axis=1))
    else:
        fixation_distances = np.array([0])  # Handle case with only 1 fixation
    distance_stats = calculate_statistics(fixation_distances)

    # 20-28. Statistics of saccade speed (distance / duration)
    if len(fixation_durations) > 1:
        saccade_speed = fixation_distances / fixation_durations.iloc[1:].values
    else:
        saccade_speed = np.array([0])
    speed_stats = calculate_statistics(saccade_speed)

    # 29-37. Statistics of left pupil diameter (LPD)
    lpd_stats = calculate_statistics(chunk['LPD'])

    # 38-46. Statistics of right pupil diameter (RPD)
    rpd_stats = calculate_statistics(chunk['RPD'])

    # 47-55. Ratio of left to right pupil diameter (LPD/RPD)
    ratio_pupil_diameter = chunk['LPD'] / chunk['RPD']
    ratio_pupil_stats = calculate_statistics(ratio_pupil_diameter)

    # 56-63. Difference between left and right pupil diameter (abs(LPD - RPD)) - 8 stats without gmean
    diff_pupil_diameter = np.abs(chunk['LPD'] - chunk['RPD'])
    diff_pupil_stats = calculate_statistics(diff_pupil_diameter, include_gmean=False)

    # Combine all statistics into one list (exactly 70 features)
    features = [
        num_fixations,
        *duration_stats.values(),
        *distance_stats.values(),
        *speed_stats.values(),
        *lpd_stats.values(),
        *rpd_stats.values(),
        *ratio_pupil_stats.values(),
        *diff_pupil_stats.values()
    ]

    # Replace NaN values with 0.0 in the features list
    features = np.array(features)  # Convert to NumPy array for NaN handling
    features[np.isnan(features)] = 0.0  # Replace NaNs with 0.0
    features = features.tolist()  # Convert back to list if needed

    return features

# Function to process the entire file and save results
def process_file(filepath):
    df = pd.read_csv(filepath)

    # Limit to first 9000 rows
    df = df.head(NUMBER_OF_MIN_ROW)

    # Split into chunks of 300 rows
    chunks = [df[i:i+SEQUENCE_LENGTH] for i in range(0, len(df), SEQUENCE_LENGTH)]

    # Process each chunk and collect statistics
    all_features = []
    for chunk in chunks:
        if len(chunk) == SEQUENCE_LENGTH:  # Only process full chunks
            features = process_chunk(chunk)
            all_features.append(features)

    # Save results to a new CSV file
    output_filepath = os.path.splitext(filepath)[0] + "_statistic.csv"

    # Use column names as 0 to 69
    columns = list(range(NUMBER_OF_FEATURES))

    result_df = pd.DataFrame(all_features, columns=columns)
    result_df.to_csv(output_filepath, index=False)
    print(f"Processed {filepath} and saved results to {output_filepath}")

# Process all files in a directory
def process_directory(directory):
    for filename in os.listdir(directory):
        if filename.endswith("gaze.csv"):
            filepath = os.path.join(directory, filename)
            process_file(filepath)

def main():
    # Example usage
    directory_path = FOLDER
    process_directory(directory_path)

main()
