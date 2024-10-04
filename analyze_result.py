import pandas as pd
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score


NUMB_OF_ITERATION = 20
NUMB_TESTER = 12
SEQUENCE_LENGTH = [450, 750, 1500, 3000]
ARCH = ["FCN", "ResNet", "ResCNN", "LSTM", "InceptionTime", "XceptionTime",]
#ARCH = ["XceptionTime"]

filepath = r"C:\Users\lehoa\OneDrive\Desktop\Tsai_model_chunk_prediction.csv"
LABEL_THRESHOLD = 0.3
analyzed_file = r"C:\Users\lehoa\OneDrive\Desktop\Tsai_model_chunk_prediction(analysed).csv"
accuracy_file = r"C:\Users\lehoa\OneDrive\Desktop\Tsai_model_chunk_prediction(accuracy).csv"
def analyze():
    df = pd.read_csv(filepath)
    df_result = pd.DataFrame(columns=['arch', 'test_user', 'iteration', 'sequence_length','groundtruth_label', 'predicted_label', 'numberof1', 'numberof0','maxsequenceof1' ])
    for arch in ARCH:
        for i in range(NUMB_OF_ITERATION):
            for sequence in SEQUENCE_LENGTH:
                for tester in range(NUMB_TESTER):
                    # Filter the dataframe based on the specific values and chunk_prediction = 1
                    filtered_df = df[
                        (df['arch'] == arch) &
                        (df['test_user'] == tester) &
                        (df['iteration'] == i+1) &
                        (df['sequence_length'] == sequence)
                        ]
                    true_label = filtered_df['label_of_test_user'].iloc[0]

                    filtered_df_1 = df[
                        (df['arch'] == arch) &
                        (df['test_user'] == tester) &
                        (df['iteration'] == i+1) &
                        (df['sequence_length'] == sequence) &
                        (df['chunk_prediction'] == 1)
                    ]
                    filtered_df_0 = df[
                        (df['arch'] == arch) &
                        (df['test_user'] == tester) &
                        (df['iteration'] == i+1) &
                        (df['sequence_length'] == sequence) &
                        (df['chunk_prediction'] == 0)
                        ]

                    # Count how many rows match the condition
                    count_1 = len(filtered_df_1)
                    count_0 = len(filtered_df_0)
                    max_consecutive_one = getMaximumsequenceofPositive(filtered_df)

                    predicted_label = 1 if count_1 / (count_1 + count_0) >= LABEL_THRESHOLD else 0

                    # Prepare data for saving
                    result_data = {
                        'arch': arch,
                        'iteration': i,
                        'sequence_length': sequence,
                        'test_user': tester,
                        'groundtruth_label': true_label,
                        'predicted_label': predicted_label,
                        'numberof1': count_1,
                        'numberof0': count_0,
                        'maxsequenceof1': max_consecutive_one
                    }
                    df_result = df_result._append(result_data, ignore_index = True)
    append_to_csv(analyzed_file,df_result)

def get_average_accuracy():
    df = pd.read_csv(filepath)
    df_result = pd.DataFrame(columns=['arch', 'iteration', 'sequence_length', 'tester_accuracy','precision','recall'])
    for arch in ARCH:
        for i in range(NUMB_OF_ITERATION):
            for sequence in SEQUENCE_LENGTH:
                groundtruth_labels = []
                predicted_labels = []
                for tester in range(NUMB_TESTER):
                    # Filter the dataframe based on the specific values and chunk_prediction = 1
                    filtered_df = df[
                        (df['arch'] == arch) &
                        (df['test_user'] == tester) &
                        (df['iteration'] == i+1) &
                        (df['sequence_length'] == sequence)
                        ]
                    groundtruth_label = filtered_df['label_of_test_user'].iloc[0]

                    filtered_df_1 = df[
                        (df['arch'] == arch) &
                        (df['test_user'] == tester) &
                        (df['iteration'] == i+1) &
                        (df['sequence_length'] == sequence) &
                        (df['chunk_prediction'] == 1)
                    ]
                    filtered_df_0 = df[
                        (df['arch'] == arch) &
                        (df['test_user'] == tester) &
                        (df['iteration'] == i+1) &
                        (df['sequence_length'] == sequence) &
                        (df['chunk_prediction'] == 0)
                        ]

                    # Count how many rows match the condition
                    count_1 = len(filtered_df_1)
                    count_0 = len(filtered_df_0)

                    predicted_label = 1 if count_1 / (count_1 + count_0) >= LABEL_THRESHOLD else 0

                    groundtruth_labels.append(groundtruth_label)
                    predicted_labels.append(predicted_label)

                accuracy_by_tester = accuracy_score(groundtruth_labels, predicted_labels)
                precision = precision_score(groundtruth_labels, predicted_labels)
                recall = recall_score(groundtruth_labels, predicted_labels)

                # Prepare data for saving
                result_data = {
                    'arch': arch,
                    'iteration': i,
                    'sequence_length': sequence,
                    'tester_accuracy': accuracy_by_tester,
                    'precision': precision,
                    'recall': recall
                }
                df_result = df_result._append(result_data, ignore_index = True)
    append_to_csv(accuracy_file,df_result)

# Create a function to append DataFrame to a CSV file
def append_to_csv(file_path, df):
    # Check if the file exists
    if not os.path.isfile(file_path):
        # File does not exist, write the DataFrame with the header
        df.to_csv(file_path, mode='w', index=False, header=True)
    else:
        # File exists, append the DataFrame without the header
        df.to_csv(file_path, mode='a', index=False, header=False)


def getMaximumsequenceofPositive(df):
    # Extract the chunk_prediction column as a list
    chunk_predictions = df['chunk_prediction'].tolist()

    # Initialize variables to track maximum consecutive 1s
    max_consecutive_ones = 0
    current_consecutive_ones = 0
    zero_allowed = True  # A flag to allow one 0 in the sequence

    # Iterate through the chunk_predictions list
    for value in chunk_predictions:
        if value == 1:
            current_consecutive_ones += 1
        elif value == 0 and zero_allowed:
            # Allow one 0 in the sequence and continue counting
            zero_allowed = False
            current_consecutive_ones += 1
        else:
            # Update max consecutive ones and reset counters
            max_consecutive_ones = max(max_consecutive_ones, current_consecutive_ones)
            current_consecutive_ones = 0
            zero_allowed = True  # Reset the flag to allow a 0 in the next sequence

    # Ensure the last sequence is considered if it ends with 1 or a single 0
    max_consecutive_ones = max(max_consecutive_ones, current_consecutive_ones)
    return max_consecutive_ones


def main():
    #analyze()
    get_average_accuracy()


main()
