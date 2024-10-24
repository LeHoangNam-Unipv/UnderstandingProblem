import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

DATA_FOLDER = r"C:\Users\lehoa\Downloads\data-20240513T042351Z-001\data"
LABEL_FILE =  r"C:\Users\lehoa\Downloads\data-20240513T042351Z-001\data\l_i_j.csv"

TOTAL_NUMB_USER = 31
TEST_SIZE = 0.2

#MIN_NUMB_ROW = None
#MIN_NUMB_ROW = 9300
MIN_NUMB_ROW = 3000  # 150 samples per second
#QUESTION = [1,2,3]
QUESTION = [1,2,3]
NUMB_QUESTION = len(QUESTION)

#FEATURES_NAME = ["FPOGX", "FPOGY", "LPCX", "LPCY" ,"LPD", "LPS", "RPCX", "RPCY" ,"RPD", "RPS"]
FEATURES_NAME = [ "LPCX", "LPCY", "LPD", "LPS", "RPCX", "RPCY", "RPD", "RPS"]

NUMBER_OF_FEATURES = len(FEATURES_NAME)  # Number of features

def extract_valid_labels(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    # Replace 'U' with 0, 'N' with 1, and 'I' with NaN to easily drop invalid labels
    df.replace({'U': 0, 'N': 1, 'I': None}, inplace=True)
    # Drop rows with any 'I' label (now NaN)
    df.dropna(inplace=True)
    # Combine 'Tester_ID' and corresponding valid labels
    valid_data = df[['User_ID', 'Question_1', 'Question_2', 'Question_3']]
    return valid_data

def get_label(valid_data, user_id, question_number):
    # Retrieve the specific label using DataFrame indexing
    result = valid_data.loc[valid_data['User_ID'] == user_id, f'Question_{question_number}']
    if not result.empty:
        return result.values[0]
    return None  # Return None if the user_id is not found


# Function to scale 3D data (list of 2D arrays)
def scale_3d_data_list(X_list, scaler):
    scaled_list = []
    for X in X_list:
        # Get the shape of the current chunk
        num_samples, num_features = X.shape
        # Scale the features (keep the sequence dimension intact)
        X_scaled = scaler.transform(X)
        # Append the scaled chunk back to the list
        scaled_list.append(X_scaled)
    return scaled_list

# Format: data_dict[(user_id, question_num)]
def load_user_data(data_folder, sequence_length):
    data_dict = {}
    user = range(0,TOTAL_NUMB_USER)

    # Loop through each question folder (P1, P2, P3)
    for question_num in QUESTION:
        folder_path = os.path.join(data_folder, f'P{question_num}')
        for user_id in user:
            # Construct the CSV file name based on user_id
            file_name = f"User {user_id}_all_gaze.csv"
            file_path = os.path.join(folder_path, file_name)

            if os.path.exists(file_path):
                # Read the CSV file into a DataFrame
                df = pd.read_csv(file_path, nrows=MIN_NUMB_ROW, usecols=FEATURES_NAME, dtype="float64")

                # Remove rows where the value in the FPOGY column is less than 0.3
                #df = df[df['FPOGY'] >= 0.3].reset_index(drop=True)

                if("FPOGY" in FEATURES_NAME and "FPOGX" in FEATURES_NAME):
                    fpogx = df["FPOGX"]
                    fpogx_diff = fpogx.diff().fillna(0)  # Take the difference over rows to capture changes
                    df["FPOGX"] = fpogx_diff

                    fpogy = df["FPOGY"]
                    fpogy_diff = fpogy.diff().fillna(0)  # Take the difference over rows to capture changes
                    df["FPOGY"] = fpogy_diff

                # Extract the changes overtime of pupil s
                if ("LPS" in FEATURES_NAME and "RPS" in FEATURES_NAME):
                    LPS = df["LPS"]
                    LPS_diff = LPS.diff().fillna(0)  # Take the difference over rows to capture changes
                    df["LPS-diff"] = LPS_diff

                    RPS = df["RPS"]
                    RPS_diff = RPS.diff().fillna(0)  # Take the difference over rows to capture changes
                    df["RPS-diff"] = RPS_diff

                # Split the DataFrame into chunks of defined sequence length
                chunks = split_into_chunks(df, sequence_length)
                # Store the list of chunks in the dictionary with (user_id, question_num) as the key
                data_dict[(user_id, question_num)] = chunks
            else:
                print(f"File not found: {file_path}")
    return data_dict

def split_into_chunks(df, sequence_length):
    # Split the DataFrame into chunks of the specified sequence length
    return [df.iloc[i:i + sequence_length].reset_index(drop=True) for i in range(0, len(df) - sequence_length + 1,
                                                                                 sequence_length)]

def balance_classes(data, label_column):
    """Balance classes in the given DataFrame."""
    class_0 = data[data[label_column] == 0]
    class_1 = data[data[label_column] == 1]
    # Find the number of samples to keep for each class
    min_samples = min(len(class_0), len(class_1))
    # Randomly sample from each class
    class_0_sampled = class_0.sample(min_samples)
    class_1_sampled = class_1.sample(min_samples)
    # Concatenate sampled data and shuffle
    balanced_data = pd.concat([class_0_sampled, class_1_sampled]).sample(frac=1)
    return balanced_data

# Format: train_data ['User_ID', 'Label', 'Source_Question']
def split_balance_class_data(valid_data, test_size=0.2):
    # Create a new DataFrame for stacking
    stacked_data = pd.DataFrame(columns=['User_ID', 'Label', 'Source_Question'])

    # Stack Questions
    for q in QUESTION:
        question = valid_data[[f'Question_{q}', 'User_ID']].copy()
        question.rename(columns={f'Question_{q}': 'Label'}, inplace=True)
        question['Source_Question'] = q
        stacked_data = pd.concat([stacked_data, question])

    stacked_data['Label'] = stacked_data['Label'].astype(int)

    # Split users into training and testing sets
    train_data, test_data = train_test_split(stacked_data, stratify=stacked_data['Label'], test_size=test_size)

    # After split training and testing set, the class is not balanced yet, so we need to do that
    # Balance the test set
    #balanced_test_data = balance_classes(test_data, 'Label')
    balanced_test_data = test_data

    # Get test unique users
    test_users = balanced_test_data['User_ID'].unique()

    # Remove test user in training data
    train_data_removed_test_user = train_data[~train_data['User_ID'].isin(test_users)]

    # Ensure balance in the train set
    #balanced_train_data = balance_classes(train_data_removed_test_user, 'Label')

    balanced_train_data = train_data_removed_test_user

    # Combine the training and testing data
    train_dataset = balanced_train_data.reset_index(drop=True)
    test_dataset = balanced_test_data.reset_index(drop=True)

    return train_dataset, test_dataset

# Format: labeled_chunks[(user_id, question_num, i)]
def assign_labels_to_chunks(user_data, valid_data):
    # Create a dictionary to store the labeled chunks
    labeled_chunks = {}

    for (user_id, question_num), chunks in user_data.items():
        # Get the corresponding label for this user and question
        if question_num == 1:
            label = valid_data.loc[valid_data['User_ID'] == user_id, 'Question_1'].values
        elif question_num == 2:
            label = valid_data.loc[valid_data['User_ID'] == user_id, 'Question_2'].values
        elif question_num == 3:
            label = valid_data.loc[valid_data['User_ID'] == user_id, 'Question_3'].values
        else:
            continue  # Skip if question_num is not valid

        # If label is found, assign it to each chunk
        if label.size > 0:
            label_value = label[0]  # Get the label value
            for i, chunk in enumerate(chunks):
                labeled_chunk = chunk.copy()
                labeled_chunk['Label'] = label_value
                # Store the chunk in the labeled_chunks dictionary
                labeled_chunks[(user_id, question_num, i)] = labeled_chunk

    return labeled_chunks

def create_X_y_from_training_data(labeled_chunks, training_data):
    # Create lists for features and labels
    X = []
    y = []

    # Iterate over each row in train_data to get user_id, label, and question
    for _, row in training_data.iterrows():
        user_id = row['User_ID']
        label = row['Label']
        source_question = row['Source_Question']

        # Get the corresponding chunks for this user and question
        question_num = int(source_question)  # Extract question number (1, 2, or 3)
        for key, chunk in labeled_chunks.items():
            if key[0] == user_id and key[1] == question_num:
                # Add the features and label to the lists
                X.append(chunk.drop(columns=['Label']).values)
                y.append(label)  # Append the label

    # Convert to numpy arrays
    #X = np.array([np.array(x) for x in X])  # List of arrays to a 3D array (num_samples, sequence_length, num_features)

    y = np.array(y)  # Convert to numpy array

    return X, y

def create_X_y_from_testing_data(labeled_chunks, test_data):
    # Create lists for features and labels
    X_test = []
    y_test = []

    # Iterate over each row in train_data to get user_id, label, and question
    for _, row in test_data.iterrows():
        X = []

        user_id = row['User_ID']
        label = row['Label']
        source_question = row['Source_Question']

        # Get the corresponding chunks for this user and question
        question_num = int(source_question)  # Extract question number (1, 2, or 3)
        for key, chunk in labeled_chunks.items():
            if key[0] == user_id and key[1] == question_num:
                # Add the features and label to the lists
                X.append(chunk.drop(columns=['Label']).values)

        X_test.append(X)
        y_test.append(label)

    # Convert to numpy arrays
    #X = np.array([np.array(x) for x in X])  # List of arrays to a 3D array (num_samples, sequence_length, num_features)

    y_test = np.array(y_test)  # Convert to numpy array

    return X_test, y_test

# In case of using all data for training, each tester might have different amount of chunks, so we need to balance
# training data

def balance_training_data(X_train, y_train):
    # Separate data by class
    class_0_indices = np.where(y_train == 0)[0]
    class_1_indices = np.where(y_train == 1)[0]

    # Find the smaller class size
    min_class_size = min(len(class_0_indices), len(class_1_indices))

    # Randomly sample the larger class to match the smaller class size
    class_0_sampled = np.random.choice(class_0_indices, min_class_size, replace=False)
    class_1_sampled = np.random.choice(class_1_indices, min_class_size, replace=False)

    # Combine the balanced indices
    balanced_indices = np.concatenate([class_0_sampled, class_1_sampled])

    # Shuffle the balanced indices
    np.random.shuffle(balanced_indices)

    # Return the balanced X_train and y_train
    X_train_balanced = [X_train[i] for i in balanced_indices]
    y_train_balanced = y_train[balanced_indices]

    X_train_balanced = np.array(X_train_balanced)
    y_train_balanced = np.array(y_train_balanced)

    return X_train_balanced, y_train_balanced

def data_generation(sequence_length):

    # Example usage:
    csv_file = LABEL_FILE
    valid_data = extract_valid_labels(csv_file)

    user_data = load_user_data(DATA_FOLDER, sequence_length)
    train_data, test_data = split_balance_class_data(valid_data, TEST_SIZE)

    labeled_chunks = assign_labels_to_chunks(user_data, valid_data)

    X_train, y_train = create_X_y_from_training_data(labeled_chunks, train_data)

    # If we use all data, we must balance the number of chunks in each class
    if (MIN_NUMB_ROW == None):
        X_train_balanced, y_train_balanced = balance_training_data(X_train, y_train)
    else:
        X_train_balanced = X_train
        y_train_balanced = y_train

    # The number of class between tester is already balanced, dont need to balance class within chunks
    X_test, y_test = create_X_y_from_testing_data(labeled_chunks, test_data)

    robust_scaler = RobustScaler()
    minmax_scaler = MinMaxScaler()

    # Use X_train to fit scaler, then transform X_train and X_test
    X_train_balanced, X_test = scaler_data(X_train_balanced, X_test, robust_scaler)
    #X_train_balanced, X_test = scaler_data(X_train_balanced, X_test, minmax_scaler)

    return X_train_balanced, y_train_balanced, X_test, y_test

def scaler_data(X_train_balanced, X_test, scaler):
    # First, flatten the 3D data into 2D for scaling
    X_train_flattened = np.concatenate([chunk for chunk in X_train_balanced], axis=0)
    # Fit the scaler on the training data
    scaler.fit(X_train_flattened)
    # Scale each chunk in X_train_balanced and X_test individually
    X_train_balanced = scale_3d_data_list(X_train_balanced, scaler)

    for i in range(len(X_test)):
        X_test[i] = scale_3d_data_list(X_test[i], scaler)

    return X_train_balanced, X_test

#def main():
    #X_train, y_train, X_test, y_test = data_generation(1500)

#main()