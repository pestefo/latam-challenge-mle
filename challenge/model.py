import pandas as pd
import numpy as np

from sklearn.utils import shuffle

from datetime import datetime
from typing import Tuple, Union, List


class DelayModel:
    def __init__(self):
        # I instantiate the object in the `fit` method, there I have
        # the `scale_pos_weight` parameter value
        self._model = None

    def preprocess(
        self, data: pd.DataFrame, target_column: str = None
    ) -> Union(Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame):
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

        # Adding aditional features
        data = self._add_additional_features(data=data)

        # Shuffling data
        training_data = shuffle(
            data[["OPERA", "MES", "TIPOVUELO", "SIGLADES", "DIANOM", "delay"]],
            random_state=111,
        )

        features = pd.concat(
            [
                pd.get_dummies(training_data["OPERA"], prefix="OPERA"),
                pd.get_dummies(training_data["TIPOVUELO"], prefix="TIPOVUELO"),
                pd.get_dummies(training_data["MES"], prefix="MES"),
            ],
            axis=1,
        )

        target = training_data["delay"]

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

        return prioritized_features, target

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """

        return

    def predict(self, features: pd.DataFrame) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.

        Returns:
            (List[int]): predicted targets.
        """
        return

    def _add_additional_features(self, data: pd.DataFrame) -> pd.DataFrame:
        data = self._add_period_day_feature(data=data)
        data = self._add_high_season(data=data)
        data = self._add_min_diff(data=data)
        data = self._add_delay_feature(data=data)

        return data

    def _add_period_day_feature(self, data: pd.DataFrame) -> pd.DataFrame:
        data["period_day"] = data["Fecha-I"].apply(self._get_period_day)
        return data

    def _add_high_season_feature(self, data: pd.DataFrame) -> pd.DataFrame:
        data["high_season"] = data["Fecha-I"].apply(self._is_high_season)
        return data

    def _add_min_diff_feature(self, data: pd.DataFrame) -> pd.DataFrame:
        data["min_diff"] = data.apply(self._get_min_diff, axis=1)
        return data

    def _add_delay_feature(self, data: pd.DataFrame) -> pd.DataFrame:
        THRESHOLD_IN_MINUTES = 15
        data["delay"] = np.where(data["min_diff"] > THRESHOLD_IN_MINUTES, 1, 0)

    def _get_period_day(self, date: datetime) -> str:
        date_time = datetime.strptime(date, "%Y-%m-%d %H:%M:%S").time()
        morning_min = datetime.strptime("05:00", "%H:%M").time()
        morning_max = datetime.strptime("11:59", "%H:%M").time()
        afternoon_min = datetime.strptime("12:00", "%H:%M").time()
        afternoon_max = datetime.strptime("18:59", "%H:%M").time()
        evening_min = datetime.strptime("19:00", "%H:%M").time()
        evening_max = datetime.strptime("23:59", "%H:%M").time()
        night_min = datetime.strptime("00:00", "%H:%M").time()
        night_max = datetime.strptime("4:59", "%H:%M").time()

        if date_time > morning_min and date_time < morning_max:
            return "mañana"
        elif date_time > afternoon_min and date_time < afternoon_max:
            return "tarde"
        elif (date_time > evening_min and date_time < evening_max) or (
            date_time > night_min and date_time < night_max
        ):
            return "noche"

    def _is_high_season(self, date: datetime) -> int:
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
            (date >= range1_min and date <= range1_max)
            or (date >= range2_min and date <= range2_max)
            or (date >= range3_min and date <= range3_max)
            or (date >= range4_min and date <= range4_max)
        ):
            return 1
        else:
            return 0

    def _get_min_diff(self, data: pd.DataFrame) -> float:
        fecha_o = datetime.strptime(data["Fecha-O"], "%Y-%m-%d %H:%M:%S")
        fecha_i = datetime.strptime(data["Fecha-I"], "%Y-%m-%d %H:%M:%S")
        min_diff = ((fecha_o - fecha_i).total_seconds()) / 60
        return min_diff
