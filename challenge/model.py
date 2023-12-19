import json
import os
from datetime import datetime
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split


class DelayModel:
    def __init__(self):
        # Constants
        self.TARGET_COLUMN = "delay"
        self.PATH_TO_TRAINED_MODEL_FILE = "models/delay_trained_model.json"
        self.PATH_TO_MODEL_SETTINGS_FILE = "settings/model_settings.json"

        # Model is initialized in the `fit` method
        self._model = None

        # Training data cache for lazy training
        self.cached_training_features = None
        self.cached_training_target = None

    def preprocess(
        self, data: pd.DataFrame, target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """

        # Adding additional features
        data_with_additiona_features = self._add_additional_features(data=data)

        # Apply one-hot encoding to categorical columns
        features = self._apply_one_hot_encoding(
            data=data_with_additiona_features
        )

        if target_column:
            target = data_with_additiona_features[[self.TARGET_COLUMN]]

            return features, target

        return features

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        # Training Settings
        training_settings = {"TEST_SIZE": 0.33, "RANDOM_STATE": 42}
        target_series = target[self.TARGET_COLUMN]

        x_train, x_test, y_train, y_test = train_test_split(
            features,
            target_series,
            test_size=training_settings["TEST_SIZE"],
            random_state=training_settings["RANDOM_STATE"],
        )

        # Calculating scale
        scale = self._calculate_scale(y_train=y_train)

        # Save model settings for future model instantiation
        model_settings = {
            "RANDOM_STATE": 1,
            "LEARNING_RATE": 0.01,
            "SCALE": scale,
        }
        self._save_model_settings(settings=model_settings)

        # Instantiating the model
        self._model = xgb.XGBClassifier(
            random_state=model_settings["RANDOM_STATE"],
            learning_rate=model_settings["LEARNING_RATE"],
            scale_pos_weight=model_settings["SCALE"],
            enable_categorical=True,
        )

        # Fitting the model
        self._model.fit(x_train, y_train)

        # Save the trained model for eventual reuse
        self._save_trained_model(model=self._model)

        return None

    def predict(self, features: pd.DataFrame) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.

        Returns:
            (List[int]): predicted targets.
        """
        model = self._get_model()

        predictions = model.predict(features)

        return predictions.tolist()

    def _add_additional_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add the additional features according to the data scientists analysis.

        Args:
            data (pd.DataFrame): raw data.

        Returns:
            (pd.DataFrame): raw data with the new columns.
        """

        data = self._add_period_day_feature(data=data)
        data = self._add_high_season_feature(data=data)
        data = self._add_min_diff_feature(data=data)
        data = self._add_delay_feature(data=data)

        return data

    def _add_period_day_feature(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add the `period_day` feature, it assigns a category "mañana", "tarde",
        or "noche" according to the value `Fecha-I`.

        Args:
            data (pd.DataFrame): raw data.

        Returns:
            (pd.DataFrame): raw data with the `period_day` column.
        """
        data["period_day"] = data["Fecha-I"].apply(self._get_period_day)
        return data

    def _add_high_season_feature(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add the `high_season` feature.

        Args:
            data (pd.DataFrame): raw data.

        Returns:
            (pd.DataFrame): raw data with the `high_season` column.
        """
        data["high_season"] = data["Fecha-I"].apply(self._is_high_season)
        return data

    def _add_min_diff_feature(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add the `min_diff` feature.

        Args:
            data (pd.DataFrame): raw data.

        Returns:
            (pd.DataFrame): raw data with the `min_diff` column.
        """

        data["min_diff"] = data.apply(self._get_min_diff, axis=1)
        return data

    def _add_delay_feature(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add the `delay` feature: it is a 1 if the `min_diff` is bigger than
        certain threshold in minutes and 0 otherwise. This threshold determines
        if an operation was delayed or not.

        Args:
            data (pd.DataFrame): raw data.

        Returns:
            (pd.DataFrame): raw data with the `delay` column.
        """

        THRESHOLD_IN_MINUTES = 15

        data["delay"] = np.where(data["min_diff"] > THRESHOLD_IN_MINUTES, 1, 0)
        return data

    def _get_period_day(self, date: str) -> str:
        """
        Return the string "mañana", "tarde" or "noche" according to the
        value `Fecha-I`.

        Args:
            date (str): date string.

        Returns:
            (str): "mañana", "tarde" or "noche" .
        """

        date_time = datetime.strptime(date, "%Y-%m-%d %H:%M:%S").time()
        morning_min = datetime.strptime("05:00", "%H:%M").time()
        morning_max = datetime.strptime("11:59", "%H:%M").time()
        afternoon_min = datetime.strptime("12:00", "%H:%M").time()
        afternoon_max = datetime.strptime("18:59", "%H:%M").time()
        evening_min = datetime.strptime("19:00", "%H:%M").time()
        evening_max = datetime.strptime("23:59", "%H:%M").time()
        night_min = datetime.strptime("00:00", "%H:%M").time()
        night_max = datetime.strptime("4:59", "%H:%M").time()

        if morning_min < date_time < morning_max:
            return "mañana"
        elif afternoon_min < date_time < afternoon_max:
            return "tarde"
        elif (evening_min < date_time < evening_max) or (
            night_min < date_time < night_max
        ):
            return "noche"

    def _is_high_season(self, date: str) -> int:
        """
        Return a 1 if the date lies between a high season date range, and
        0 otherwise.

        Args:
            date (str): date string.

        Returns:
            (int): 0 or 1.
        """

        date_year = int(date.split("-")[0])
        date = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
        range1_min = datetime.strptime("15-Dec", "%d-%b").replace(
            year=date_year
        )
        range1_max = datetime.strptime("31-Dec", "%d-%b").replace(
            year=date_year
        )
        range2_min = datetime.strptime("1-Jan", "%d-%b").replace(
            year=date_year
        )
        range2_max = datetime.strptime("3-Mar", "%d-%b").replace(
            year=date_year
        )
        range3_min = datetime.strptime("15-Jul", "%d-%b").replace(
            year=date_year
        )
        range3_max = datetime.strptime("31-Jul", "%d-%b").replace(
            year=date_year
        )
        range4_min = datetime.strptime("11-Sep", "%d-%b").replace(
            year=date_year
        )
        range4_max = datetime.strptime("30-Sep", "%d-%b").replace(
            year=date_year
        )

        if (
            (range1_min <= date <= range1_max)
            or (range2_min <= date <= range2_max)
            or (range3_min <= date <= range3_max)
            or (range4_min <= date <= range4_max)
        ):
            return 1
        else:
            return 0

    def _get_min_diff(self, data: pd.DataFrame) -> float:
        """
        Return the minimal difference (in seconds) between the scheduled date
        and time of the flight, and actual date and time of flight operation.

        Args:
            date (pd.DataFrame): raw data.

        Returns:
            (float): the difference in secondes between these two instants.
        """

        fecha_o = datetime.strptime(data["Fecha-O"], "%Y-%m-%d %H:%M:%S")
        fecha_i = datetime.strptime(data["Fecha-I"], "%Y-%m-%d %H:%M:%S")
        min_diff = ((fecha_o - fecha_i).total_seconds()) / 60
        return min_diff

    def _calculate_scale(self, y_train: pd.DataFrame) -> float:
        """
        Calculate the scale parameter

        Args:
            y_train (pd.DataFrame): y training data.

        Returns:
            (float): scale value.
        """
        n_y0 = len(y_train[y_train == 0])
        n_y1 = len(y_train[y_train == 1])
        return n_y0 / n_y1

    def _apply_one_hot_encoding(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        As the most relevant features are categorical, it is needed to apply
        one-hot encoding. This method applies it to the `data` dataframe by
        leveraging the use of the `top_10_features` obtained through the data
        scientist's analysis.

        Args:
            date (pd.DataFrame): raw data.

        Returns:
            (pd.DataFrame): one-hot encoded features.
        """

        # Prioritized features discovered during the analysis
        top_10_features = [
            "OPERA_Latin American Wings",
            "MES_7",
            "MES_10",
            "OPERA_Grupo LATAM",
            "MES_12",
            "TIPOVUELO_I",
            "MES_4",
            "MES_11",
            "OPERA_Sky Airline",
            "OPERA_Copa Air",
        ]

        # One-hot encoding the relevant categorical columns
        opera_dummies = pd.get_dummies(data["OPERA"], prefix="OPERA")
        tipo_vuelo_dummies = pd.get_dummies(
            data["TIPOVUELO"], prefix="TIPOVUELO"
        )
        mes_dummies = pd.get_dummies(data["MES"], prefix="MES")

        # Ensure presence of `top_10_features` in one-hot encoded DataFrames,
        # fill with zeros if missing
        for column in top_10_features:
            if column not in opera_dummies.columns:
                opera_dummies[column] = 0
            if column not in tipo_vuelo_dummies.columns:
                tipo_vuelo_dummies[column] = 0
            if column not in mes_dummies.columns:
                mes_dummies[column] = 0

        # Create the feature dataset by concatenated the one-hot encoded
        # dataframes
        features = pd.concat(
            [
                opera_dummies[
                    [
                        column
                        for column in top_10_features
                        if "OPERA_" in column
                    ]
                ],
                tipo_vuelo_dummies[
                    [
                        column
                        for column in top_10_features
                        if "TIPOVUELO_" in column
                    ]
                ],
                mes_dummies[
                    [column for column in top_10_features if "MES_" in column]
                ],
            ],
            axis=1,
        )

        return features

    def _save_model_settings(self, settings: dict) -> None:
        """
        Writes the model_settings into a file

        Args:
            settings (dict): dictionary with the model settings.
        """

        with open(self.PATH_TO_MODEL_SETTINGS_FILE, "w") as fp:
            json.dump(settings, fp)

        return None

    def _get_model_settings(self) -> dict:
        """
        Return the settings to instantiate the model.

        Returns:
            (dict): dictionary with the model settings.
        """

        with open(self.PATH_TO_MODEL_SETTINGS_FILE, "r") as fp:
            return json.load(fp)

    def _save_trained_model(self, model: xgb.XGBClassifier) -> None:
        """
        Writes the model_settings into a file

        Args:
            settings (dict): dictionary with the model settings.
        """
        model.save_model(self.PATH_TO_TRAINED_MODEL_FILE)

    def _get_model(self) -> xgb.XGBClassifier:
        """
        Return the model object already trained saved as a model cache.

        Returns:
            (dict): dictionary with the model settings.

        Side effects:
            In case the model instance has not been initialized and there
            is model cache the ModelNotTrainedException exception is raised.
        """

        if self._model is None:
            model_settings = self._get_model_settings()

            if os.path.exists(self.PATH_TO_TRAINED_MODEL_FILE):
                try:
                    model = xgb.XGBClassifier(
                        random_state=model_settings["RANDOM_STATE"],
                        learning_rate=model_settings["LEARNING_RATE"],
                        scale_pos_weight=model_settings["SCALE"],
                        enable_categorical=True,
                    )

                    model.load_model(self.PATH_TO_TRAINED_MODEL_FILE)

                    self._model = model
                except Exception:
                    raise ModelNotTrainedException()

        return self._model


class ModelNotTrainedException(Exception):
    pass
