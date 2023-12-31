from datetime import datetime
from typing import List

from pydantic import BaseModel, validator

VALID_FLIGHT_OPERATORS = [
    "American Airlines",
    "Air Canada",
    "Air France",
    "Aeromexico",
    "Aerolineas Argentinas",
    "Austral",
    "Avianca",
    "Alitalia",
    "British Airways",
    "Copa Air",
    "Delta Air",
    "Gol Trans",
    "Iberia",
    "K.L.M.",
    "Qantas Airways",
    "United Airlines",
    "Grupo LATAM",
    "Sky Airline",
    "Latin American Wings",
    "Plus Ultra Lineas Aereas",
    "JetSmart SPA",
    "Oceanair Linhas Aereas",
    "Lacsa",
]


class FlightModel(BaseModel):
    OPERA: str
    TIPOVUELO: str
    MES: int
    Fecha_I: str
    Fecha_O: str

    @validator("OPERA")
    def valid_flight_operator(cls, operator):
        if operator not in VALID_FLIGHT_OPERATORS:
            raise ValueError("Must provide a valid `OPERA`")
        return operator

    @validator("TIPOVUELO")
    def vuelo_is_either_N_or_I(cls, tipo_vuelo_):
        if tipo_vuelo_ not in ("N", "I"):
            raise ValueError("Must provide a valid `TIPOVUELO`")
        return tipo_vuelo_

    @validator("MES")
    def valid_month_number(cls, month_number):
        if month_number not in range(1, 13):
            raise ValueError("Must provide a valid MES")
        return month_number

    @validator("Fecha_I")
    def fecha_i_validator(cls, fecha):
        return cls.fecha_in_isoformat(fecha)

    @validator("Fecha_O")
    def fecha_o_validator(cls, fecha):
        return cls.fecha_in_isoformat(fecha)

    def fecha_in_isoformat(fecha):
        try:
            datetime.fromisoformat(fecha)
            return fecha

        except ValueError:
            raise ValueError("Must provide a valid `Fecha_I` or `Fecha_O`")


class FlightListModel(BaseModel):
    flights: List[FlightModel]


class DelayPredictionsModel(BaseModel):
    predictions: List[int]
