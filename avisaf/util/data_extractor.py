#!/usr/bin/env python3
"""The data extractor module is responsible for getting raw text data and
training data used by other modules.
"""

import pandas as pd
from sys import stderr
import json
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from avisaf_ner.avisaf.training.training_data_creator import build_feature_matrices_from_texts
import sklearn.metrics as metrics
import numpy as np


def get_entities(entities_file_path: Path = Path('entities_labels.json').resolve()):
    """Function which reads given JSON file supposed to contain the list of user
    defined entity labels.

    :type entities_file_path: Path
    :param entities_file_path: The path to the JSON file containing the list
        of entity labels.

    :return: Returns the list of available entity labels.
    """
    entities_file_path = entities_file_path.resolve()

    with entities_file_path.open(mode='r') as entities_file:
        return json.load(entities_file)


def get_training_data(training_data_file_path: Path):
    """Function which reads given JSON file supposed to contain the training data.
    The training data are supposed to be a list of (text, annotations) tuples.

    :type training_data_file_path: Path
    :param training_data_file_path: The path to the JSON file containing the
        training data.

    :return: Returns the JSON list of (text, annotations) tuples.
    """

    if not training_data_file_path.is_absolute():
        training_data_file_path = training_data_file_path.resolve()

    with training_data_file_path.open(mode='r') as tr_data_file:
        return json.load(tr_data_file)


def get_narratives(file_path: Path, lines_count: int = -1, start_index: int = 0):
    """Function responsible for reading raw csv file containing the original
    safety reports from the ASRS database.

    :type lines_count: int
    :param lines_count: Number of lines to be read.
    :type file_path: Path
    :param file_path: The path to the csv file containing the texts.
    :type start_index: int
    :param start_index: Number indicating the index of the first text to be
        returned.

    :return: Returns a python generator object of all texts.
    """

    file_path = str(file_path) if file_path.is_absolute() else str(file_path.resolve())

    report_df = pd.read_csv(
        file_path,
        skip_blank_lines=True,
        index_col=0,
        header=[0, 1]
    )
    report_df.columns = report_df.columns.map('_'.join)

    try:
        narratives1 = report_df['Report 1_Narrative'].values.tolist()
        calls1 = report_df['Report 1_Callback'].values.tolist()
        narratives2 = report_df['Report 2_Narrative'].values.tolist()
        calls2 = report_df['Report 2_Callback'].values.tolist()

    except KeyError:
        print('No such key was found', file=stderr)
        return None

    length = len(narratives1)
    lists = [narratives1, calls1, narratives2, calls2]

    end_index = start_index + lines_count

    result = []
    for index in range(start_index, length):
        if lines_count != -1 and index >= end_index:
            break
        result.append(' '.join([str(lst[index]) for lst in lists if str(lst[index]) != 'nan']))

    return result


def find_file_by_path(file_path: [Path, str], max_iterations: int = 5):
    """

    :param max_iterations:
    :param file_path:
    :return:
    """

    current_dir = Path().resolve()

    for iteration in range(max_iterations):  # Searching at most max_iterations times.
        if current_dir.parent is None or current_dir.parent == current_dir:
            break

        requested_file = current_dir.joinpath(Path(file_path)).resolve()

        if requested_file.exists():
            return requested_file
        else:
            current_dir = current_dir.parent

    return None


def extract_data_from_csv_columns(file_path: [Path, list], field_name: [str, list], lines_count: int = -1, start_index: int = 0):
    """

    :param field_name:
    :param file_path:
    :param lines_count:
    :param start_index:
    :return:
    """

    # Normalizing the type of arguments to fit the rest of the method
    file_paths = file_path if type(file_path) is list else [file_path]
    field_names = field_name if type(field_name) is list else [field_name]

    label_data_dict = {}
    for a_field_name in field_names:
        extracted_values = []
        for a_file_path in file_paths:
            requested_file = find_file_by_path(a_file_path)
            if requested_file is None:
                print(f'The file given by "{a_file_path}" path was not found in the given range.', file=stderr)
                # Ignoring the file with current file path
                continue

            csv_dataframe = pd.read_csv(
                requested_file,
                skip_blank_lines=True,
                header=[0, 1],
            )
            csv_dataframe.columns = csv_dataframe.columns.map('_'.join)
            csv_dataframe = csv_dataframe.replace(np.nan, "", regex=True)

            try:
                extracted_values_collection = csv_dataframe[a_field_name].values.tolist()
            except KeyError:
                print(f'"{field_name} is not a correct field name. Please make sure the column name is in format "FirstLineTitle_SecondLineTitle"', file=stderr)
                continue

            length = len(extracted_values_collection)
            end_index = start_index + lines_count if lines_count != -1 else length
            extracted_values += extracted_values_collection[start_index: end_index]  # Getting only the desired subset of extracted data

        label_data_dict[a_field_name] = extracted_values

    return label_data_dict


def main():
    # TODO: Vytvor main funkciu ktorá použije argparser na výber funkcie
    return 1


if __name__ == '__main__':
    paths = [Path(arg) for arg in sys.argv[1:-1]]
    result1 = extract_data_from_csv_columns(
        paths,
        ['Report 1_Narrative', 'Events_Detector']
    )
    data, target = build_feature_matrices_from_texts(
        result1['Report 1_Narrative'],
        result1['Events_Detector'],
        target_label_filter=['Person Flight Crew', 'Person Air Traffic Control']
    )

    train_data, test_data, train_target, test_target = train_test_split(
        data, target,
        test_size=0.15,
        random_state=1998
    )

    print(f'train data shape: {train_data.shape}')
    print(f'train target shape: {train_target.shape}')
    print(f'test data shape: {test_data.shape}')
    print(f'test target shape: {test_target.shape}')

    # classifier = MLPClassifier(hidden_layer_sizes=(256, 64, 32))
    # model = classifier.fit(train_data, train_target)

    classifier = KNeighborsClassifier(weights='uniform')

    # classifier = SVC()
    model = classifier.fit(train_data, train_target)

    predictions = model.predict(test_data)

    print(predictions)
    print(f'Accuracy: {metrics.accuracy_score(test_target, predictions) * 100}')
    print(f'F1-score: {metrics.f1_score(test_target, predictions) * 100}')
    print('==================================')
    predictions = np.zeros(test_target.shape)
    print(f'Accuracy predicting always 0: {metrics.accuracy_score(test_target, predictions) * 100}')
    print(f'F1-score: {metrics.f1_score(test_target, predictions) * 100}')
    print('==================================')
    predictions = np.ones(test_target.shape)
    print(f'Accuracy predicting always 1: {metrics.accuracy_score(test_target, predictions) * 100}')
    print(f'F1-score: {metrics.f1_score(test_target, predictions) * 100}')

