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
from sklearn.base import clone
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

    def __init__(self, models, vectorizer=None, normalized: bool = True, deviation_rate: float = 0.0, parameters=None):

        if parameters is None:
            parameters = dict()
        self._models = models  # Model(s) to be used for evaluation

        if parameters.get("model_params") is not None:
            for model in self._models:
                for param, value in parameters["model_params"].items():
                    setattr(model, param, value)

        self._normalize = normalized
        self._preprocessor = ASRSReportDataPreprocessor(vectorizer)
        self._vectorizer = vectorizer
        self._deviation_rate = deviation_rate

        try:
            self._encodings = parameters["encodings"]  # "int: label" dictionary of possible classes
            self._trained_filtered_labels = parameters["trained_labels"]
        except AttributeError:
            raise ValueError("Corrupted model parameters")

        assert len(self._models) == len(self._trained_filtered_labels.keys())

    """def predict_report_class(self, texts_paths: list, label_to_test: str = None, label_filter: list = None):
        
        labels_to_test = self._trained_filtered_labels.keys()
        labels_filters = self._trained_filtered_labels.values()

        if label_to_test is not None:
            labels_to_test = [label_to_test]
            
            if label_filter is None:
                if self._trained_filtered_labels.get(label_to_test):
                    labels_filters = self._trained_filtered_labels[label_to_test]
                else:
                    labels_filters = None
            else:
                labels_filters = [label_filter]
            
        for lbl, fltr in zip(labels_to_test, labels_filters):

            test_data, test_target = self._preprocessor.vectorize_texts(
                texts_paths,
                lbl,
                train=False,
                label_values_filter=fltr,
                normalize=self._normalize
            )

            logger.info(self._preprocessor.get_data_distribution(test_target)[1])

            logger.info(f'Test data shape: {test_data.shape}')
            predictions = self.predict(test_data, None)

        return predictions, test_target"""

    @staticmethod
    def predict(test_data, model, predict_proba: bool = True):
        if model is None:
            raise ValueError('A model needs to be trained or loaded first to perform predictions.')

        logger.info(f'Probability predictions made using model: {model}')
        predictions = model.predict_proba(test_data) if predict_proba else model.predict(test_data)

        return predictions

    def _decode_prediction(self, prediction: int):
        if not len(self._encodings):
            raise ValueError('Train a model to get an non-empty encoding.')

        decoded_label = self._encodings.get(prediction)

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
        prediction = self._models.predict(vectorized_text)
        predicted_label = self._decode_prediction(prediction)

        # TODO: For a given text, the classifier returns a dictionary containing field name as key and its predicted value
        return predicted_label


class ASRSReportClassificationEvaluator:

    def __init__(self):
        pass

    @staticmethod
    def evaluate(predictions: list, test_targets):

        ensemble = ""
        if len(predictions) > 1:
            logger.debug(f"{len(predictions)} models ensembling")
            ensemble = f"(ensemble of {len(predictions)} models)"

        predictions = np.mean(predictions, axis=0)

        for predictions_distribution, class_targets in zip(predictions, test_targets):
            class_predictions = np.argmax(predictions_distribution, axis=1)
            unique_predictions_count = np.unique(class_targets).shape[0]
            avg = 'binary' if unique_predictions_count == 2 else 'micro'

            print('==============================================')
            print('Confusion matrix: number [i,j] indicates the number of observations of class i which were predicted to be in class j')
            print(metrics.confusion_matrix(class_targets, class_predictions))
            if ensemble:
                print(ensemble)
            print('Model Based Accuracy: {:.2f}'.format(metrics.accuracy_score(class_targets, class_predictions) * 100))
            print('Model Based Precision: {:.2f}'.format(metrics.precision_score(class_targets, class_predictions) * 100))
            print('Model Based Recall: {:.2f}'.format(metrics.recall_score(class_targets, class_predictions) * 100))
            print('Model Based F1-score: {:.2f}'.format(metrics.f1_score(class_targets, class_predictions, average=avg) * 100))
            print('==============================================')
            for unique_prediction in range(unique_predictions_count):
                mockup_predictions = np.full(class_targets.shape, unique_prediction)
                print(f'Accuracy predicting always {unique_prediction}: {metrics.accuracy_score(class_targets, mockup_predictions) * 100}')
                print(f'F1-score: {metrics.f1_score(class_targets, mockup_predictions, average=avg) * 100}')
                print(f'Model Based Precision: {metrics.precision_score(class_targets, mockup_predictions, zero_division=1) * 100}')
                print(f'Model Based Recall: {metrics.recall_score(class_targets, mockup_predictions) * 100}')
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

    def __init__(self, models=None, parameters: dict = None, algorithm=None, normalized: bool = True, vectorizer=None, deviation_rate: float = 0.0):

        # TODO: Initialize an empty model for each field classifier
        def set_classification_algorithm(classification_algorithm: str):
            available_classifiers = {
                'mlp': MLPClassifier(
                    hidden_layer_sizes=(256, 32),
                    alpha=0.005,
                    batch_size=256,
                    learning_rate='adaptive',
                    learning_rate_init=0.005,
                    random_state=6240,
                    verbose=True,
                    early_stopping=True
                ),
                'svm': SVC(probability=True),
                'tree': DecisionTreeClassifier(criterion='entropy', max_features=10000),
                'forest': RandomForestClassifier(
                    n_estimators=200,
                    criterion='entropy',
                    min_samples_split=32,
                    max_features=20000,
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

        if not models:
            self._classifier = set_classification_algorithm(algorithm)
            self._models = []
            self._deviation_rate = 0.0
            self._encodings = dict()
            self._model_params = self._classifier.get_params()
            self._params = dict()
            self._algorithm = algorithm
            self._trained_filtered_labels = dict()
            self._trained_texts = []
        else:
            try:
                self._classifier = models[0]
                self._models = models
                self._deviation_rate = deviation_rate
                self._params = parameters
                encodings = {}
                for label, encoding in parameters["encodings"].items():
                    encodings.update({label: {int(key): value for key, value in encoding.items()}})
                self._encodings = encodings
                self._model_params = parameters["model_params"]
                self._algorithm = parameters["algorithm"]
                self._trained_filtered_labels = parameters["trained_labels"]
                self._trained_texts = parameters["trained_texts"]
            except AttributeError:
                raise ValueError("Corrupted parameters.json file")

        assert len(self._models) == len(self._encodings.keys())

        if self._classifier is not None and parameters.get("model_params") is not None:
            for param, value in parameters["model_params"].items():
                try:
                    setattr(self._classifier, param, value)
                except AttributeError:
                    logging.warning(f"Trying to set a non-existing attribute { param } with value { value }")

    def train_report_classification(self, texts_paths: list, label_to_train: str, label_filter: list = None, mode: str = "dev"):

        if mode not in ["train", "dev", "test"]:
            raise ValueError("Unsupported argument")

        if mode == "train":
            self._trained_texts += texts_paths

        labels_to_train = list(self._trained_filtered_labels.keys())
        labels_filters = list(self._trained_filtered_labels.values())

        if label_to_train is not None:
            labels_to_train = [label_to_train]

            if label_filter is None:
                if self._trained_filtered_labels.get(label_to_train):
                    labels_filters = self._trained_filtered_labels[label_to_train]
                    filter_update = labels_filters
                else:
                    labels_filters = None
                    filter_update = []
            else:
                labels_filters = [label_filter]
                filter_update = label_filter

            if mode == "train":
                self._trained_filtered_labels.update({label_to_train: filter_update})

        if not labels_to_train:
            raise ValueError("Nothing to train - please make sure at least one category is specified.")

        assert len(labels_to_train) == len(labels_filters)

        labels_predictions, labels_targets = [], []
        for lbl, fltr in zip(labels_to_train, labels_filters):

            data, target = self._preprocessor.vectorize_texts(
                texts_paths,
                lbl,
                train=True,
                label_values_filter=fltr,
                normalize=self._normalize
            )

            logger.debug(f'{ mode } data shape: {data.shape}')
            logger.debug(self._preprocessor.get_data_distribution(target)[1])

            if mode == "train":
                self._classifier = clone(self._classifier)
                # encoding is available only after texts vectorization
                self._encodings.update({lbl: self._preprocessor.get_encoding()})

                logger.info(self._classifier)

                self._classifier.fit(data, target)
                self._params = {
                    "algorithm": self._algorithm,
                    "encodings": self._encodings,
                    "model_params": self._model_params,
                    "trained_labels": self._trained_filtered_labels,
                    "trained_texts": self._trained_texts,
                    "vectorizer_params": self._preprocessor.vectorizer.get_params()
                }

                logging.info(f"MODEL: {self._classifier}")
                # self.save_model(self._model)
                self._models.append(self._classifier)

            predictions = ASRSReportClassificationPredictor.predict(data, self._classifier)
            labels_predictions.append(predictions)
            labels_targets.append(target)
            # ASRSReportClassificationEvaluator.evaluate([[predictions]], [target])

        if mode == "train":
            self.save_model(self._models)

        return labels_predictions, labels_targets

    def save_model(self, models_to_save: list):
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
            logger.info(f'Saving {len(models_to_save)} model(s): {models_to_save}')
            logger.debug(f'Saving vectorizer: {self._preprocessor.vectorizer}')
            pickle.dump((models_to_save, self._preprocessor.vectorizer), model_file)

        with open(Path("classifiers", model_dir_name, 'parameters.json'), 'w', encoding="utf-8") as params_file:
            logger.info(f'Saving parameters [encoding, model parameters, train_texts_paths, trained_labels, label_filter]')
            json.dump(self._params, params_file, indent=4)
            # self._params = parameters


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
                    models, vectorizer = pickle.load(model_file)

                with open(Path(models_dir_paths[idx], 'parameters.json'), 'r') as params_file:
                    parameters = json.load(params_file)
            else:
                models = None
                vectorizer = None
                parameters = None

            classifier = ASRSReportClassificationTrainer(
                models=models,
                algorithm=algorithm,
                vectorizer=vectorizer,
                parameters=parameters,
                normalized=normalize,
                deviation_rate=deviation_rate
            )
            classifier.train_report_classification(texts_paths, label, label_filter, mode=mode)
    else:
        logging.debug(f'Testing on { "normalized " if normalize else "" }{ mode }')

        if models_dir_paths is None:
            raise ValueError("The path to the model cannot be null for testing")

        models_predictions = []
        test_targets = None
        for model_dir_path in models_dir_paths:

            with lzma.open(Path(model_dir_path, 'classifier.model'), 'rb') as model_file:
                models, vectorizer = pickle.load(model_file)
                logging.debug(f"Loaded {len(models)} models")

            with open(Path(model_dir_path, 'parameters.json'), 'r') as params_file:
                parameters = json.load(params_file)

            predictor = ASRSReportClassificationTrainer(
                models=models,
                parameters=parameters,
                vectorizer=vectorizer,
                normalized=normalize,
                deviation_rate=deviation_rate
            )

            if not texts_paths:
                texts_paths = [f'../ASRS/ASRS_{ mode }.csv']

            predictions, targets = predictor.train_report_classification(texts_paths, label, label_filter, mode=mode)
            models_predictions.append(predictions)
            if test_targets is None:
                test_targets = targets
        if plot:
            ASRSReportClassificationEvaluator.plot(models_predictions, test_targets)

        ASRSReportClassificationEvaluator.evaluate(models_predictions, test_targets)
