import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from tsai.all import *
from fastai.callback.all import EarlyStoppingCallback, SaveModelCallback
import logging
# Import the Adam optimizer
from fastai.optimizer import Adam
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score

from datasetCreation import data_generation
#from datasetCreationWithSingleSlideforTesting import data_generation

parent_directory = os.getcwd()

NUMBER_OF_EPOCH = 200
VALIDATION_SIZE = 0.2
BATCH_SIZE = 64
PATIENCE = 10  # Early stopping

now = datetime.now()
date_string = now.strftime("%d-%m-%Y-%H-%M-%S")

#SEQUENCE_LENGTH = [450, 750, 1500, 3000]
SEQUENCE_LENGTH = [750,1500]
NUMBER_OF_IMPLEMENTATION = 20
#NUMBER_OF_FEATURES = 10
NUMBER_OF_FEATURES = 10

def build_models():
    # Define the architectures and their hyperparameters
    archs = [
        (FCN, {}),
        #(ResNet, {}),
        #(xresnet1d34, {}),  # NOT ok
        #(ResCNN, {}),
        #(LSTM, {'n_layers':1, 'bidirectional': False, 'fc_dropout':0.2, 'rnn_dropout':0.2}), # no
        #(LSTM, {'n_layers':2, 'bidirectional': False, 'fc_dropout':0.2, 'rnn_dropout':0.2}), # no
        #(LSTM, {'c_out':2, 'n_layers':3, 'hidden_size':32, 'bidirectional': False, 'fc_dropout':0.2, 'rnn_dropout':0.2}), # no
        #(LSTM, {'n_layers':1, 'bidirectional': True, 'fc_dropout':0.2, 'rnn_dropout':0.2}), # no
        #(LSTM, {'n_layers':2, 'bidirectional': True}), # no
        #(LSTM, {'c_out':2, 'n_layers':3, 'hidden_size':32, 'bidirectional': True, 'fc_dropout':0.2, 'rnn_dropout':0.2}),
        #(LSTM_FCN, {}), # Note: Long training time
        #(LSTM_FCN, {'shuffle': False}),
        #(InceptionTime, {}),
        #(XceptionTime, {}),
        #(OmniScaleCNN, {}), # Not useful
        #(mWDN, {'levels': 4})  # NOT ok
    ]
    return archs


def return_predicted_label_and_majority(pred_labels):
    # Count the number of 1s
    count_1 = np.sum(pred_labels)
    # Count the number of 0s
    count_0 = len(pred_labels) - count_1
    # Determine which value is higher and calculate the probability
    if count_1 >= count_0:
        higher_value = 1
        higher_count = count_1
    else:
        higher_value = 0
        higher_count = count_0

    probability_higher_value = higher_count / len(pred_labels)
    return higher_value, probability_higher_value

def train_model(archs, dls, validation_data_X, validation_data_y):
    results = pd.DataFrame(columns=['arch', 'hyperparams', 'total params', 'model_saved_name', 'valid loss', 'precision', 'recall', 'valid accuracy', 'true_validation_labels', 'predicted_label','predicted_label_majority','accuracy_by_tester'])
    chunk_prediction = pd.DataFrame(columns=['arch', 'test_user', 'label_of_test_user', 'chunk_prediction', 'chunk_prediction_confidence'])
    for i, (arch, k) in enumerate(archs):
        model = create_model(arch, dls=dls, **k)
        print(model.__class__.__name__)
        model_saved_name = str(datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
        # Initialize the EarlyStoppingCallback
        save_callback = SaveModelCallback(monitor='valid_loss', comp=None, fname=model_saved_name, every_epoch=False,
                                          at_end=False, with_opt=False, reset_on_fit=True)
        early_stopping  = EarlyStoppingCallback(patience = PATIENCE)

        loss = CrossEntropyLossFlat()
        learn = Learner(dls, model, metrics=[Precision(), Recall(), accuracy], loss_func=loss, opt_func=Adam)
        #learn.fit(NUMBER_OF_EPOCH, cbs=[save_callback, early_stopping])
        learn.fit(NUMBER_OF_EPOCH, cbs=[early_stopping])
        #start = time.time()
        #learn.fit_one_cycle(NUMBER_OF_EPOCH, cbs=[save_callback, early_stopping], lr_max=1e-3)

        #elapsed = time.time() - start
        #vals = learn.recorder.values[-1]

        #results.sort_values(by='accuracy', ascending=False, kind='stable', ignore_index=True, inplace=True)
        #clear_output()
        #display(results)

        #print(f"validation_data_y: {str(validation_data_y)}")

        predicted_labels = []
        label_majority = []
        # Validation
        for tester_index in range(len(validation_data_X)):
            tester_features = validation_data_X[tester_index]
            true_label = validation_data_y[tester_index]
            #print(true_label)
            #print(tester_features.shape)
            preds = []
            for sample in tester_features:

                # Convert to tensor if necessary
                sample_reshaped = np.expand_dims(sample, axis=0)
                #sample_tensor = torch.tensor(sample_reshaped, dtype=torch.float32)
                #batch_tfms = TSStandardize(use_single_batch=True)
                # Create a TSDataset
                #sample_ds = TSDataset(sample_tensor)
                # Create a DataLoader from the TSDataset
                #dls = TSDataLoader(sample_ds, bs=1, batch_tfms=batch_tfms)
                # Perform prediction

                test_probas, test_targets, test_preds = learn.get_X_preds(sample_reshaped)
                df = {'arch': arch.__name__, 'test_user': tester_index, 'label_of_test_user': true_label, 'chunk_prediction': test_preds[1], 'chunk_prediction_confidence': str(test_probas)}
                chunk_prediction = chunk_prediction._append(df, ignore_index = True)
                #print(f"prediction: {test_preds}")
                preds.append(int(test_preds[1]))
            #print(preds)
            # Convert predictions to class labels if needed
            #pred_labels = np.argmax(preds, axis=1)
            pred_label, majority = return_predicted_label_and_majority(preds)
            predicted_labels.append(int(pred_label))
            label_majority.append(majority)

        # Calculate accuracy
        accuracy_by_tester = accuracy_score(validation_data_y, predicted_labels)
        #print(str(predicted_label_with_confidence))

        validation = learn.validate()
        print(validation)
        logging.info(validation)

        results.loc[i] = [arch.__name__, k, count_parameters(model), model_saved_name, validation[0], validation[1],
                          validation[2], validation[3], str(validation_data_y), str(predicted_labels),str(label_majority), accuracy_by_tester]

    return results, chunk_prediction


def implementation(sequence):
    X, y, validation_data_X, validation_data_y = data_generation(sequence)

    # Split the data into training and validation sets
    splits = get_splits(y, valid_size=VALIDATION_SIZE, shuffle=False, show_plot=False)

    # Create the TSDatasets
    tfms = [None, [Categorize()]]

    batch_tfms = TSStandardize(use_single_batch=True)

    dsets = TSDatasets(X, y, tfms=tfms, splits=splits)
    #dsets_test = TSDatasets(X_test, y_test, tfms=tfms)

    # Create the TSDataLoaders
    dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[BATCH_SIZE, BATCH_SIZE * 2], batch_tfms=batch_tfms)

    archs = build_models()

    results, chunk_prediction = train_model(archs, dls, validation_data_X, validation_data_y)

    return results, chunk_prediction

def save_chunk_prediction_to_csv(file_path, chunk_prediction):
    f = os.path.join(parent_directory, "prediction", file_path)
    append_to_csv(f, chunk_prediction)

def save_result_to_csv(file_path, results):
    f = os.path.join(parent_directory, "result", file_path)
    # Append new DataFrame to the existing CSV file
    append_to_csv(f, results)


# Create a function to append DataFrame to a CSV file
def append_to_csv(file_path, df):
    # Check if the file exists
    if not os.path.isfile(file_path):
        # File does not exist, write the DataFrame with the header
        df.to_csv(file_path, mode='w', index=False, header=True)
    else:
        # File exists, append the DataFrame without the header
        df.to_csv(file_path, mode='a', index=False, header=False)

def main():
    logging.basicConfig(filename=os.path.join(parent_directory, 'log', f'{date_string}.log'),
                        format='%(asctime)s %(message)s', encoding='utf-8', level=logging.INFO)
    #logging.info(f"Slide: {PROCESSED_FOLDER}")
    logging.info(f"NUMBER_OF_EPOCH: {NUMBER_OF_EPOCH}")
    logging.info(f"VALIDATION_SIZE: {VALIDATION_SIZE}")
    logging.info(f"SEQUENCE_LENGTH: {SEQUENCE_LENGTH}")
    logging.info(f"PATIENCE: {PATIENCE}")
    logging.info('Started')
    try:
        for sequence in SEQUENCE_LENGTH:
            for i in range(NUMBER_OF_IMPLEMENTATION):
                results, chunk_prediction = implementation(sequence)
                results["iteration"] = i+1
                results["sequence_length"] = sequence
                chunk_prediction["iteration"] = i+1
                chunk_prediction["sequence_length"] = sequence/150
                save_result_to_csv(f"{str(date_string)}.csv", results)
                save_chunk_prediction_to_csv(f"{str(date_string)}.csv", chunk_prediction)
    except Exception as error:
        logging.error(error)
    logging.info('Finished')

main()