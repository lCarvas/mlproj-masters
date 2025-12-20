"""mlproj-web REST API."""

from __future__ import annotations

import logging
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, cast

import polars as pl
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import DataConversionWarning
from sklearn.pipeline import FunctionTransformer, Pipeline

from funcs.data_import import import_data
from funcs.pipeline import build_preprocessing_pipeline

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    import numpy as np
    from numpy.typing import NDArray
    from pandas import DataFrame, Series

warnings.filterwarnings("ignore", category=DataConversionWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None]:
    """Handle FastAPI startup and shutdown events."""
    # Startup events.
    for handler in logging.root.handlers:
        logging.root.removeHandler(handler)
    yield
    # Shutdown events.


app: FastAPI = FastAPI(lifespan=lifespan)

static_path: Path = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_path), name="static")

templates_path: Path = Path(__file__).parent / "templates"
templates: Jinja2Templates = Jinja2Templates(directory=templates_path)


train_df: pl.DataFrame = import_data("data/train.csv")
x_train: pl.DataFrame = train_df.drop("price", "paintQuality%")
y_train: pl.DataFrame = train_df.select("price")
preproc_pipeline: Pipeline = build_preprocessing_pipeline(
    metric_features=[
        "mileage",
        "tax",
        "mpg",
        "engineSize",
        "previousOwners",
    ],
    bool_features=[
        "hasDamage",
    ],
    categorical_features=[
        "Brand",
        "model",
        "year",
        "transmission",
        "fuelType",
    ],
    thresholds={
        "mileage": {"lower": 1, "upper": None},
        "tax": {"lower": 0, "upper": 400},
        "mpg": {"lower": 20, "upper": 200},
        "engineSize": {"lower": 1.0, "upper": 6.0},
        "previousOwners": {"lower": 0, "upper": None},
        "year": {"lower": 0, "upper": 2020},
    },
    winsorize=False,
    unneeded_float_features=[
        "year",
        "mileage",
        "tax",
        "previousOwners",
        "hasDamage",
    ],
    scaling_exclude_selector=("_", "hasDamage"),
    columns_to_coalesce=(
        "Brand",
        "model",
        "transmission",
        "fuelType",
        "year",
    ),
)


estimator = Pipeline(
    steps=[
        ("preprocessing", preproc_pipeline),
        (
            "feature_selection",
            FunctionTransformer(
                lambda x: x.select(
                    [
                        "Brand_audi",
                        "Brand_bmw",
                        "Brand_ford",
                        "Brand_hyundai",
                        "Brand_mercedes",
                        "Brand_unknown",
                        "Brand_opel",
                        "Brand_skoda",
                        "Brand_toyota",
                        "Brand_vw",
                        "carID",
                        "engineSize",
                        "fuelType_diesel",
                        "fuelType_hybrid",
                        "fuelType_unknown",
                        "fuelType_other",
                        "mileage",
                        "model_1 series",
                        "model_2 series",
                        "model_3 series",
                        "model_4 series",
                        "model_5 series",
                        "model_6 series",
                        "model_7 series",
                        "model_8 series",
                        "model_a class",
                        "model_a1",
                        "model_a3",
                        "model_a4",
                        "model_a5",
                        "model_a6",
                        "model_a7",
                        "model_a8",
                        "model_adam",
                        "model_amarok",
                        "model_arteon",
                        "model_astra",
                        "model_auris",
                        "model_aygo",
                        "model_b class",
                        "model_beetle",
                        "model_c class",
                        "model_c-hr",
                        "model_california",
                        "model_caravelle",
                        "model_cl class",
                        "model_cla class",
                        "model_cls class",
                        "model_corsa",
                        "model_e class",
                        "model_edge",
                        "model_fabia",
                        "model_fiesta",
                        "model_focus",
                        "model_g class",
                        "model_gla class",
                        "model_glc class",
                        "model_gle class",
                        "model_gls class",
                        "model_golf",
                        "model_grandland x",
                        "model_gt86",
                        "model_i10",
                        "model_i20",
                        "model_i8",
                        "model_i800",
                        "model_insignia",
                        "model_ix20",
                        "model_ka+",
                        "model_kamiq",
                        "model_karoq",
                        "model_kodiaq",
                        "model_kuga",
                        "model_land cruiser",
                        "model_m class",
                        "model_m3",
                        "model_m4",
                        "model_m5",
                        "model_mokka x",
                        "model_mondeo",
                        "model_mustang",
                        "model_octavia",
                        "model_passat",
                        "model_polo",
                        "model_prius",
                        "model_puma",
                        "model_q3",
                        "model_q5",
                        "model_q7",
                        "model_q8",
                        "model_r8",
                        "model_rav4",
                        "model_rs3",
                        "model_rs4",
                        "model_rs5",
                        "model_rs6",
                        "model_s class",
                        "model_s4",
                        "model_s8",
                        "model_santa fe",
                        "model_sharan",
                        "model_shuttle",
                        "model_sl class",
                        "model_slk",
                        "model_sq7",
                        "model_superb",
                        "model_supra",
                        "model_t-cross",
                        "model_t-roc",
                        "model_tiguan",
                        "model_tourneo custom",
                        "model_tt",
                        "model_tucson",
                        "model_unknown",
                        "model_up",
                        "model_v class",
                        "model_viva",
                        "model_x-class",
                        "model_x1",
                        "model_x3",
                        "model_x4",
                        "model_x5",
                        "model_x6",
                        "model_x7",
                        "model_yaris",
                        "model_z4",
                        "mpg",
                        "previousOwners",
                        "tax",
                        "transmission_automatic",
                        "transmission_manual",
                        "transmission_semi-auto",
                        "transmission_unknown",
                        "year_1998",
                        "year_1999",
                        "year_2001",
                        "year_2002",
                        "year_2003",
                        "year_2004",
                        "year_2005",
                        "year_2006",
                        "year_2007",
                        "year_2008",
                        "year_2009",
                        "year_2010",
                        "year_2011",
                        "year_2012",
                        "year_2013",
                        "year_2014",
                        "year_2015",
                        "year_2016",
                        "year_2017",
                        "year_2018",
                        "year_2019",
                        "year_2020",
                        "year_unknown",
                    ]
                )
            ),
        ),
        ("to_pandas", FunctionTransformer(lambda x: x.to_pandas())),
        (
            "regressor",
            RandomForestRegressor(
                max_depth=18,
                max_features=0.8512243795321489,
                max_leaf_nodes=200,
                min_impurity_decrease=0.0009513328212059685,
                min_samples_leaf=1,
                min_samples_split=50,
                n_estimators=661,
                n_jobs=-1,
                random_state=42,
            ),
        ),
    ]
)

estimator.fit(cast("DataFrame", x_train), cast("Series", y_train))

MODELS: dict[str, tuple[str, ...]] = {
    "audi": (
        "a1",
        "a2",
        "a3",
        "a4",
        "a5",
        "a6",
        "a7",
        "a8",
        "q2",
        "q3",
        "q5",
        "q7",
        "q8",
        "r8",
        "rs3",
        "rs4",
        "rs5",
        "rs6",
        "rs7",
        "s3",
        "s4",
        "s5",
        "s8",
        "sq5",
        "sq7",
        "tt",
    ),
    "bmw": (
        "1 series",
        "2 series",
        "3 series",
        "4 series",
        "5 series",
        "6 series",
        "7 series",
        "8 series",
        "i3",
        "i8",
        "m2",
        "m3",
        "m4",
        "m5",
        "m6",
        "x1",
        "x2",
        "x3",
        "x4",
        "x5",
        "x6",
        "x7",
        "z3",
        "z4",
    ),
    "ford": (
        "b-max",
        "c-max",
        "ecosport",
        "edge",
        "escort",
        "fiesta",
        "focus",
        "fusion",
        "galaxy",
        "grand c-max",
        "grand tourneo connect",
        "ka",
        "ka+",
        "kuga",
        "mondeo",
        "mustang",
        "puma",
        "ranger",
        "s-max",
        "streetka",
        "tourneo connect",
        "tourneo custom",
        "transit tourneo",
    ),
    "hyundai": (
        "accent",
        "amica",
        "getz",
        "i10",
        "i20",
        "i30",
        "i40",
        "i800",
        "ioniq",
        "ix20",
        "ix35",
        "kona",
        "santa fe",
        "terracan",
        "tucson",
        "veloster",
    ),
    "mercedes": (
        "180",
        "200",
        "220",
        "230",
        "a class",
        "b class",
        "c class",
        "cl class",
        "cla class",
        "clc class",
        "clk",
        "cls class",
        "e class",
        "g class",
        "gl class",
        "gla class",
        "glb class",
        "glc class",
        "gle class",
        "gls class",
        "m class",
        "r class",
        "s class",
        "sl class",
        "slk",
        "v class",
        "x-class",
    ),
    "skoda": (
        "citigo",
        "fabia",
        "kamiq",
        "karoq",
        "kodiaq",
        "octavia",
        "rapid",
        "roomster",
        "scala",
        "superb",
        "yeti",
        "yeti outdoor",
    ),
    "opel": (
        "adam",
        "agila",
        "ampera",
        "antara",
        "astra",
        "cascada",
        "combo life",
        "corsa",
        "crossland x",
        "grandland x",
        "gtc",
        "insignia",
        "kadjar",
        "meriva",
        "mokka",
        "mokka x",
        "tigra",
        "vectra",
        "viva",
        "vivaro",
        "zafira",
        "zafira tourer",
    ),
    "vw": (
        "amarok",
        "arteon",
        "beetle",
        "caddy",
        "caddy life",
        "caddy maxi",
        "caddy maxi life",
        "california",
        "caravelle",
        "cc",
        "eos",
        "fox",
        "golf",
        "golf sv",
        "jetta",
        "passat",
        "polo",
        "scirocco",
        "sharan",
        "shuttle",
        "t-cross",
        "t-roc",
        "tiguan",
        "tiguan allspace",
        "touareg",
        "touran",
        "up",
    ),
    "toyota": (
        "auris",
        "avensis",
        "aygo",
        "c-hr",
        "camry",
        "corolla",
        "gt86",
        "hilux",
        "iq",
        "land cruiser",
        "prius",
        "proace verso",
        "rav4",
        "supra",
        "urban cruiser",
        "verso",
        "verso-s",
        "yaris",
    ),
}

TRANSMISSIONS: list[str] = [
    "Automatic",
    "Manual",
    "Semi-Auto",
    "Other",
    "Unknown",
]

FUEL_TYPES: list[str] = ["Petrol", "Diesel", "Electric", "Hybrid", "Other"]


class CarData(BaseModel):
    """Car data model for form submission."""

    brand: str
    model: str
    year: int = Field(gt=0)
    transmission: str
    mileage: int = Field(ge=0)
    fuel_type: str
    tax: int = Field(ge=0)
    mpg: float = Field(gt=0)
    engine_size: float = Field(gt=0)
    previous_owners: int = Field(ge=0)
    has_damage: bool


@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request) -> HTMLResponse:
    """Serve the car details form page."""
    return templates.TemplateResponse(
        request,
        "car_form.html",
        {
            "request": request,
            "brands": tuple(MODELS.keys()),
            "transmissions": TRANSMISSIONS,
            "fuel_types": FUEL_TYPES,
        },
    )


@app.get("/api/models/{brand}")
async def get_models_by_brand(brand: str) -> dict[str, list[str]]:
    """Get models for a specific brand."""
    brand_lower = brand.lower()
    if brand not in MODELS:
        raise HTTPException(
            status_code=404, detail=f"Brand '{brand}' not found"
        )
    return {"models": list(MODELS[brand_lower])}


@app.post("/submit")
async def submit_form(car_data: CarData) -> dict[str, str | float]:
    """Handle car form submission."""
    prediction: NDArray[np.float64] = cast(
        "NDArray[np.float64]",
        estimator.predict(
            pl.DataFrame(
                {
                    "Brand": [car_data.brand],
                    "model": [car_data.model],
                    "year": [car_data.year],
                    "transmission": [car_data.transmission],
                    "mileage": [car_data.mileage],
                    "fuelType": [car_data.fuel_type],
                    "tax": [car_data.tax],
                    "mpg": [car_data.mpg],
                    "engineSize": [car_data.engine_size],
                    "previousOwners": [car_data.previous_owners],
                    "hasDamage": [car_data.has_damage],
                },
                schema={
                    "Brand": pl.String,
                    "model": pl.String,
                    "year": pl.Float64,
                    "transmission": pl.String,
                    "mileage": pl.Float64,
                    "fuelType": pl.String,
                    "tax": pl.Float64,
                    "mpg": pl.Float64,
                    "engineSize": pl.Float64,
                    "previousOwners": pl.Float64,
                    "hasDamage": pl.Float64,
                },
            )
        ),
    )

    return {
        "message": "Price prediction completed successfully",
        "price": round(prediction[0], 2),
    }
