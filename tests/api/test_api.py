import unittest

from fastapi.testclient import TestClient

from challenge import app


class TestBatchPipeline(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_should_get_predict(self):
        data = {
            "flights": [
                {
                    "OPERA": "Aerolineas Argentinas",
                    "TIPOVUELO": "N",
                    "MES": 3,
                    "Fecha_I": "2017-03-05 09:50:00",
                    "Fecha_O": "2017-03-05 10:50:00"
                }
            ]
        }
        # when("xgboost.XGBClassifier").predict(ANY).thenReturn(np.array([0])) # change this line to the model of chosing
        response = self.client.post("/predict", json=data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"predict": [0]})

    def test_should_failed_unknown_column_1(self):
        data = {
            "flights": [
                {
                    "OPERA": "Aerolineas Argentinas",
                    "TIPOVUELO": "N",
                    "MES": 13,
                    "Fecha_I": "2017-01-05 09:50:00",
                    "Fecha_O": "2017-01-05 10:50:00"
                }
            ]
        }
        # when("xgboost.XGBClassifier").predict(ANY).thenReturn(np.array([0]))# change this line to the model of chosing
        response = self.client.post("/predict", json=data)
        self.assertEqual(response.status_code, 400)

    def test_should_failed_unknown_column_2(self):
        data = {
            "flights": [
                {
                    "OPERA": "Aerolineas Argentinas",
                    "TIPOVUELO": "O",
                    "MES": 13,
                    "Fecha_I": "2017-01-05 09:50:00",
                    "Fecha_O": "2017-01-05 10:50:00"
                }
            ]
        }
        # when("xgboost.XGBClassifier").predict(ANY).thenReturn(np.array([0]))# change this line to the model of chosing
        response = self.client.post("/predict", json=data)
        self.assertEqual(response.status_code, 400)

    def test_should_failed_unknown_column_3(self):
        data = {
            "flights": [
                {
                    "OPERA": "Argentinas",
                    "TIPOVUELO": "O",
                    "MES": 13,
                    "Fecha_I": "2017-01-05 09:50:00",
                    "Fecha_O": "2017-01-05 10:50:00"
                }
            ]
        }
        # when("xgboost.XGBClassifier").predict(ANY).thenReturn(np.array([0]))
        response = self.client.post("/predict", json=data)
        self.assertEqual(response.status_code, 400)

    # additional test
    def test_should_failed_unknown_fecha(self):
        data = {
            "flights": [
                {
                    "OPERA": "Argentinas",
                    "TIPOVUELO": "O",
                    "MES": 13,
                    "Fecha_I": "AAAA2017-01-05 09:50:00",
                    "Fecha_O": "2017-01-05 10:50:00"
                }
            ]
        }
        # when("xgboost.XGBClassifier").predict(ANY).thenReturn(np.array([0]))
        response = self.client.post("/predict", json=data)
        self.assertEqual(response.status_code, 400)

    def test_should_get_predict_more_than_one_flight(self):
        data = {
            "flights": [
                {
                    "OPERA": "Aerolineas Argentinas",
                    "TIPOVUELO": "N",
                    "MES": 3,
                    "Fecha_I": "2019-03-05 07:50:00",
                    "Fecha_O": "2019-03-05 07:52:00"
                },
                {
                    "OPERA": "Avianca",
                    "TIPOVUELO": "I",
                    "MES": 6,
                    "Fecha_I": "2017-06-05 09:50:00",
                    "Fecha_O": "2017-06-05 10:50:00"
                }
            ]
        }
        # when("xgboost.XGBClassifier").predict(ANY).thenReturn(np.array([0])) # change this line to the model of chosing
        response = self.client.post("/predict", json=data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"predict": [0, 1]})
