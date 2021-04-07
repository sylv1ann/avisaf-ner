#!/usr/bin/env python3
"""

"""

import lzma
import json
import pickle
import logging
import numpy as np
from re import sub
from datetime import datetime
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from pathlib import Path
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import SVC

from avisaf.training.training_data_creator import ASRSReportDataPreprocessor
logger = logging.getLogger(str(__file__))

logging.basicConfig(
    level=logging.DEBUG,
    format=f'[%(levelname)s - %(asctime)s]: %(message)s'
)


class ASRSReportClassificationPredictor:

    def __init__(self, model=None, vectorizer=None, normalized: bool = True, deviation_rate: float = 0.0, parameters=None):

        if parameters is None:
            parameters = dict()
        self._model = model  # Model(s) to be used for evaluation

        if parameters.get("model_params") is not None:
            for param, value in parameters["model_params"].items():
                setattr(self._model, param, value)

        self._normalize = normalized
        self._preprocessor = ASRSReportDataPreprocessor(vectorizer)
        self._vectorizer = vectorizer
        self._deviation_rate = deviation_rate

        try:
            self._encoding = parameters["encoding"]  # "int: label" dictionary of possible classes
            self._trained_label = list(parameters["trained_label"].keys())[0]
            self._trained_filter = parameters["trained_label"][self._trained_label]
        except AttributeError:
            raise ValueError("Corrupted model parameters")

    def predict_report_class(self, texts_paths: list, label_to_test: str = None, label_filter: list = None):

        if label_to_test is None and self._trained_label is not None:
            label_to_test = self._trained_label

        if label_filter is None and self._trained_filter is not None:
            label_filter = self._trained_filter

        test_data, test_target = self._preprocessor.vectorize_texts(
            texts_paths,
            label_to_test,
            train=False,
            label_values_filter=label_filter,
            normalize=self._normalize
        )

        logger.info(self._preprocessor.get_data_distribution(test_target)[1])

        logger.info(f'Test data shape: {test_data.shape}')
        predictions = self.predict_proba(test_data)
        return predictions, test_target

    def predict(self, test_data, model_to_use=None):
        model = self._model if model_to_use is None else model_to_use

        if model is None:
            raise ValueError('A model needs to be trained or loaded first to perform predictions.')

        logger.info(f'Predictions made using model: {model}')
        predictions = model.predict(test_data)

        return predictions

    def predict_proba(self, test_data, model_to_use=None):
        model = self._model if model_to_use is None else model_to_use

        if model is None:
            raise ValueError('A model needs to be trained or loaded first to perform predictions.')

        logger.info(f'Probability predictions made using model: {model}')
        predictions = model.predict_proba(test_data)

        return predictions

    def _decode_prediction(self, prediction: int):
        if not len(self._encoding):
            raise ValueError('Train a model to get an non-empty encoding.')

        decoded_label = self._encoding.get(prediction)

        if decoded_label is None:
            raise ValueError(f'Encoding with value "{prediction}" does not exist.')

        return decoded_label

    def decode_predictions(self, predictions: list):
        if predictions is None:
            raise TypeError('Predictions have to be made first')

        vectorized = np.vectorize(self._decode_prediction)
        decoded_labels = vectorized(predictions)

        return decoded_labels

    def label_text(self, text):
        """

        :param text:
        :return:
        """
        if self._vectorizer is None:
            raise ValueError('A model needs to be trained or loaded first to be able to transform texts.')

        vectorized_text = self._vectorizer.transform(text)
        prediction = self._model.predict(vectorized_text)
        predicted_label = self._decode_prediction(prediction)

        # TODO: For a given text, the classifier returns a dictionary containing field name as key and its predicted value
        return predicted_label


class ASRSReportClassificationEvaluator:

    def __init__(self):
        pass

    @staticmethod
    def evaluate(predictions: list, test_target):

        ensemble = ""
        if len(predictions) > 1:
            logger.debug(f"{len(predictions)} models ensembling")
            ensemble = f"(ensemble of {len(predictions)} models)"

        predictions = np.argmax(np.mean(predictions, axis=0), axis=1)

        unique_predictions_count = np.unique(test_target).shape[0]
        avg = 'binary' if unique_predictions_count == 2 else 'micro'

        # sample_weights = np.ndarray((predictions.shape[0]))

        # for idx, pred in enumerate(predictions):
        #    sample_weights[idx] = 1 if pred == 1 else 2

        print('==============================================')
        print('Confusion matrix: number [i,j] indicates the number of observations of class i which were predicted to be in class j')
        print(metrics.confusion_matrix(test_target, predictions))
        if ensemble:
            print(ensemble)
        print('Model Based Accuracy: {:.2f}'.format(metrics.accuracy_score(test_target, predictions) * 100))
        print('Model Based Precision: {:.2f}'.format(metrics.precision_score(test_target, predictions, average=avg) * 100))
        print('Model Based Recall: {:.2f}'.format(metrics.recall_score(test_target, predictions, average=avg) * 100))
        print('Model Based F1-score: {:.2f}'.format(metrics.f1_score(test_target, predictions, average=avg) * 100))
        print('==============================================')
        for unique_prediction in range(unique_predictions_count):
            predictions = np.full(test_target.shape, unique_prediction)
            print(f'Accuracy predicting always {unique_prediction}: {metrics.accuracy_score(test_target, predictions) * 100}')
            print(f'F1-score: {metrics.f1_score(test_target, predictions, average=avg) * 100}')
            print(f'Model Based Precision: {metrics.precision_score(test_target, predictions, zero_division=1, average=avg) * 100}')
            print(f'Model Based Recall: {metrics.recall_score(test_target, predictions, average=avg) * 100}')
            print('==============================================')

    @staticmethod
    def plot(probability_predictions, test_target):
        preds = np.mean(probability_predictions, axis=0)[:, 1]

        fpr, tpr, threshold = metrics.roc_curve(test_target, preds)
        roc_auc = metrics.auc(fpr, tpr)
        # prec, recall, thr = metrics.precision_recall_curve(test_target, preds)

        plt.title('ROC Curve')
        plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()


class ASRSReportClassificationTrainer:

    def __init__(self, model=None, parameters: dict = None, algorithm=None, normalized: bool = True, vectorizer=None, deviation_rate: float = 0.0):

        # TODO: Initialize an empty model for each field classifier
        def set_classification_algorithm(classification_algorithm: str):
            available_classifiers = {
                'mlp': MLPClassifier(
                    hidden_layer_sizes=(128, 64),
                    alpha=0.005,
                    batch_size=128,
                    learning_rate='adaptive',
                    learning_rate_init=0.005,
                    random_state=6240,
                    verbose=True,
                    max_iter=80,
                    early_stopping=True,
                    n_iter_no_change=30
                ),
                'svm': SVC(probability=True),
                'tree': DecisionTreeClassifier(criterion='entropy', max_features=10000),
                'forest': RandomForestClassifier(
                    n_estimators=150,
                    criterion='entropy',
                    min_samples_split=32,
                    n_jobs=2,
                    # max_features=20000,
                    verbose=5
                ),
                'knn': KNeighborsClassifier(n_neighbors=15),
                'gauss': GaussianNB(),
                'mnb': MultinomialNB(),
                'bernoulli': BernoulliNB()
            }

            # Setting a default classifier value
            _classifier = available_classifiers['knn']

            if available_classifiers.get(classification_algorithm) is not None:
                _classifier = available_classifiers[classification_algorithm]

            return _classifier

        if parameters is None:
            parameters = dict()

        self._normalize = normalized
        self._preprocessor = ASRSReportDataPreprocessor(vectorizer)

        if model is None:
            self._model = set_classification_algorithm(algorithm)
            self._deviation_rate = 0.0
            self._encoding = dict()
            self._model_params = self._model.get_params()
            self._params = dict()
            self._algorithm = algorithm
            self._trained_filtered_labels = dict()
            self._trained_texts = []
        else:
            try:
                self._model = model
                self._deviation_rate = deviation_rate
                self._params = parameters
                self._encoding = {int(key): value for key, value in parameters["encoding"].items()}
                self._model_params = parameters["model_params"]
                self._algorithm = parameters["algorithm"]
                self._trained_filtered_labels = parameters["trained_label"]
                self._trained_texts = parameters["trained_texts"]
            except AttributeError:
                raise ValueError("Corrupted parameters.json file")

        if self._model is not None and parameters.get("model_params") is not None:
            for param, value in parameters["model_params"].items():
                try:
                    setattr(self._model, param, value)
                except AttributeError:
                    logging.warning(f"Trying to set a non-existing attribute { param } with value { value }")

    def train_report_classification(self, texts_paths: list, label_to_train: str, label_filter: list = None):

        self._trained_texts += texts_paths

        if label_to_train is not None:
            x = {label_to_train: label_filter if label_filter else []}
            self._trained_filtered_labels.update(x)
        else:
            # TODO: Add support for extracting multiple labels at the same time
            label_to_train = list(self._trained_filtered_labels.keys())[0]
            label_filter = self._trained_filtered_labels[label_to_train]

        train_data, train_target = self._preprocessor.vectorize_texts(
            texts_paths,
            label_to_train,
            train=True,
            label_values_filter=label_filter,
            normalize=self._normalize
        )

        # encoding is available only after texts vectorization
        self._encoding.update(self._preprocessor.get_encoding())

        logger.debug(self._preprocessor.get_data_distribution(train_target)[1])

        logger.debug(f'Train data shape: {train_data.shape}')
        logger.info(self._model)

        self._model = self._model.fit(train_data, train_target)

        logging.info(f"MODEL: {self._model}")

        self._params = {
                "algorithm": self._algorithm,
                "encoding": self._encoding,
                "model_params": self._model_params,
                "trained_label": self._trained_filtered_labels,
                "trained_texts": self._trained_texts,
                "vectorizer_params": self._preprocessor.vectorizer.get_params()
            }

        train_data_evaluator = ASRSReportClassificationPredictor(
            model=self._model,
            parameters=self._params,
            vectorizer=self._preprocessor.vectorizer,
            normalized=self._normalize,
        )
        predictions = train_data_evaluator.predict_proba(train_data)
        ASRSReportClassificationEvaluator.evaluate([predictions], train_target)
        self.save_model(self._model)

    def save_model(self, model_to_save):
        model_dir_name = "asrs_classifier-{}-{}-{}".format(
            self._algorithm,
            datetime.now().strftime("%Y%m%d_%H%M%S"),
            ",".join(("{}_{}".format(sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(self._model_params.items()))).replace(" ", "_", -1)
        )

        model_dir_name = model_dir_name[:100]

        if self._normalize:
            model_dir_name += ',norm'

        Path("classifiers").mkdir(exist_ok=True)
        Path("classifiers", model_dir_name).mkdir(exist_ok=False)
        with lzma.open(Path("classifiers", model_dir_name, 'classifier.model'), 'wb') as model_file:
            logger.info(f'Saving model: {model_to_save}')
            logger.debug(f'Saving vectorizer: {self._preprocessor.vectorizer}')
            pickle.dump((model_to_save, self._preprocessor.vectorizer), model_file)

        with open(Path("classifiers", model_dir_name, 'parameters.json'), 'w', encoding="utf-8") as params_file:
            logger.info(f'Saving parameters [encoding, model parameters, train_texts_paths, trained_label, label_filter]')
            json.dump(self._params, params_file, indent=4)


def launch_classification(models_dir_paths: list, texts_paths: list, label: str, label_filter: list, algorithm: str, normalize: bool, mode: str, plot: bool):

    deviation_rate = np.random.uniform(low=0.95, high=1.05, size=None) if normalize else None  # 5% of maximum deviation between classes
    if mode == 'train':
        logging.debug('Training')

        if models_dir_paths is None:
            models_dir_paths = []

        min_iterations = max(len(models_dir_paths), 1)  # we want to iterate through all given models or once if no model was given
        for idx in range(min_iterations):

            if models_dir_paths:
                with lzma.open(Path(models_dir_paths[idx], 'classifier.model'), 'rb') as model_file:
                    model, vectorizer = pickle.load(model_file)

                with open(Path(models_dir_paths[idx], 'parameters.json'), 'r') as params_file:
                    parameters = json.load(params_file)
            else:
                model = None
                vectorizer = None
                parameters = None

            classifier = ASRSReportClassificationTrainer(
                model=model,
                algorithm=algorithm,
                vectorizer=vectorizer,
                parameters=parameters,
                normalized=normalize,
                deviation_rate=deviation_rate
            )
            classifier.train_report_classification(texts_paths, label, label_filter)
    else:
        logging.debug(f'Testing on { "normalized " if normalize else "" }{ mode }')

        if models_dir_paths is None:
            raise ValueError("The path to the model cannot be null for testing")

        models_predictions = []
        test_targets = None
        for model_dir_path in models_dir_paths:

            with lzma.open(Path(model_dir_path, 'classifier.model'), 'rb') as model_file:
                model, vectorizer = pickle.load(model_file)

            with open(Path(model_dir_path, 'parameters.json'), 'r') as params_file:
                parameters = json.load(params_file)

            predictor = ASRSReportClassificationPredictor(
                model=model,
                parameters=parameters,
                vectorizer=vectorizer,
                normalized=normalize,
                deviation_rate=deviation_rate
            )

            if not texts_paths:
                texts_paths = [f'../ASRS/ASRS_{ mode }.csv']

            predictions, targets = predictor.predict_report_class(texts_paths, label, label_filter)
            models_predictions.append(predictions)
            if test_targets is None:
                test_targets = targets
        if plot:
            ASRSReportClassificationEvaluator.plot(models_predictions, test_targets)
        ASRSReportClassificationEvaluator.evaluate(models_predictions, test_targets)

    """
    pre = ASRSReportDataPreprocessor()

    _, targets = pre.vectorize_texts(texts_paths, label, train=True, label_values_filter=label_filter)
    d = pre.get_data_distribution(targets)
    print(d)
    print(f'Different values: {len(d[0])}')
    print(f'Percentages: {np.around(d[1] * 100, decimals=2)}')
    print(f'Elements: {np.sum(d[0])}')"""
