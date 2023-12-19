import fastapi
import pandas as pd
from fastapi.exceptions import ValidationError
from fastapi.responses import JSONResponse

from .entities import DelayPredictionsModel, FlightListModel
from .model import DelayModel

app = fastapi.FastAPI()
model = DelayModel()


@app.exception_handler(ValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=fastapi.status.HTTP_400_BAD_REQUEST,
        content="Must provide a valid request payload",
    )


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {"status": "OK"}


@app.post("/predict", status_code=200)
async def post_predict(request: dict) -> dict:
    def flights_to_features(flight_list: FlightListModel) -> pd.DataFrame:
        features: pd.DataFrame = pd.concat(
            [pd.json_normalize(dict(flight)) for flight in flight_list.flights]
        )
        features.rename(
            columns={"Fecha_I": "Fecha-I", "Fecha_O": "Fecha-O"}, inplace=True
        )
        return features

    def get_predictions(flight_list: FlightListModel) -> DelayPredictionsModel:
        raw_features: pd.DataFrame = flights_to_features(
            flight_list=flight_list
        )
        features = model.preprocess(data=raw_features)
        raw_predictions = model.predict(features=features)
        return DelayPredictionsModel(predictions=raw_predictions)

    delay_predictions = get_predictions(
        flight_list=FlightListModel(flights=request["flights"])
    )

    return JSONResponse(content={"predict": delay_predictions.predictions})
