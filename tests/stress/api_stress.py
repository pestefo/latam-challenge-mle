from locust import HttpUser, task


class StressUser(HttpUser):
    @task
    def predict_argentinas(self):
        self.client.post(
            "/predict",
            json={
                "flights": [
                    {
                        "OPERA": "Aerolineas Argentinas",
                        "TIPOVUELO": "N",
                        "MES": 3,
                        "Fecha_I": "2017-03-05 09:50:00",
                        "Fecha_O": "2017-03-05 10:50:00",
                    }
                ]
            },
        )

    @task
    def predict_latam(self):
        self.client.post(
            "/predict",
            json={
                "flights": [
                    {
                        "OPERA": "Grupo LATAM",
                        "TIPOVUELO": "N",
                        "MES": 3,
                        "Fecha_I": "2017-03-05 11:50:00",
                        "Fecha_O": "2017-03-05 14:50:00",
                    }
                ]
            },
        )
