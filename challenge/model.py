import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from datetime import datetime
from typing import Tuple, Union, List


class DelayModel:
    def __init__(self):
        self.target = "delay"

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
        data = self._add_additional_features(data=data)

        # Shuffling data
        training_data = shuffle(
            data[["OPERA", "MES", "TIPOVUELO", "SIGLADES", "DIANOM", "delay"]],
            random_state=111,
        )

        # Setting up the features and target training data
        features = pd.concat(
            [
                pd.get_dummies(training_data["OPERA"], prefix="OPERA"),
                pd.get_dummies(training_data["TIPOVUELO"], prefix="TIPOVUELO"),
                pd.get_dummies(training_data["MES"], prefix="MES"),
            ],
            axis=1,
        )

        target = training_data[[self.target]]

        # Setting the top 10 most important features
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

        prioritized_features = features[top_10_features]

        # Set a cache of the data given for training the model,
        # so we can train it later if necessary
        self._set_training_data_cache(features=prioritized_features, target=target)

        if target_column is None:
            return prioritized_features

        return prioritized_features, target

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """

        # Settings
        TEST_SIZE = 0.33
        RANDOM_STATE = 42
        LEARNING_RATE = 0.01

        x_train, x_test, y_train, y_test = train_test_split(
            features, target, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

        # Calculating scale
        n_y0 = len(y_train[y_train == 0])
        n_y1 = len(y_train[y_train == 1])
        scale = n_y0 / n_y1

        # Instantiating the model
        self._model = xgb.XGBClassifier(
            random_state=1, learning_rate=LEARNING_RATE, scale_pos_weight=scale
        )

        # Fitting the model
        self._model.fit(x_train, y_train)

        return None

    def predict(self, features: pd.DataFrame) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.

        Returns:
            (List[int]): predicted targets.
        """
        if self._model is None:
            cached_features, cached_target = self._get_training_data_cache()
            self.fit(features=cached_features, target=cached_target)

        return self._model.predict(features).tolist()

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
        range1_min = datetime.strptime("15-Dec", "%d-%b").replace(year=date_year)
        range1_max = datetime.strptime("31-Dec", "%d-%b").replace(year=date_year)
        range2_min = datetime.strptime("1-Jan", "%d-%b").replace(year=date_year)
        range2_max = datetime.strptime("3-Mar", "%d-%b").replace(year=date_year)
        range3_min = datetime.strptime("15-Jul", "%d-%b").replace(year=date_year)
        range3_max = datetime.strptime("31-Jul", "%d-%b").replace(year=date_year)
        range4_min = datetime.strptime("11-Sep", "%d-%b").replace(year=date_year)
        range4_max = datetime.strptime("30-Sep", "%d-%b").replace(year=date_year)

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

    def _set_training_data_cache(
        self, features: pd.DataFrame, target: pd.DataFrame
    ) -> None:
        """
        Saves a cache for the features and target. They are neccesary in case
        the model is required to predict and it have not been trained yet.

        Args:
            features (pd.DataFrame): features values.
            target (pd.DataFrame): target values.
        """
        self.cached_training_features = features
        self.cached_training_target = target

    def _get_training_data_cache(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Provides the cache of training data cached values.

        Returns:
            (pd.DataFrame): features values.
            (pd.DataFrame): target values.
        """
        if self.cached_training_features is None or self.cached_training_target is None:
            raise ModelNotTrainedException()

        return self.cached_training_features, self.cached_training_target


class ModelNotTrainedException(Exception):
    pass
