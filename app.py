"""FastAPI application for Motherboard Health AI inference."""

from __future__ import annotations

from typing import Any

from predict import MotherboardHealthPredictor

try:
    from fastapi import FastAPI
    from pydantic import BaseModel, Field
except ImportError:
    FastAPI = None
    BaseModel = object
    Field = None


if FastAPI is not None:
    app = FastAPI(
        title="Motherboard Health AI",
        docs_url=None,
        redoc_url=None,
        openapi_url=None
    )
    predictor = MotherboardHealthPredictor()

    class MotherboardTelemetry(BaseModel):
        """Prediction payload schema."""

        ModelName: str = Field(..., examples=["Dell Inspiron 6880"])
        CPUUsage: float = Field(..., ge=0, le=100)
        RAMUsage: float = Field(..., ge=0, le=100)
        Temperature: float = Field(..., ge=0)
        Voltage: float = Field(..., ge=0)
        DiskUsage: float = Field(..., ge=0, le=100)
        FanSpeed: float = Field(..., ge=0)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/predict")
    def predict(payload: MotherboardTelemetry) -> dict[str, Any]:
        return predictor.predict(payload.model_dump())
else:
    app = None


if __name__ == "__main__":
    if FastAPI is None:
        print("FastAPI is not installed. Run `python -m pip install -r requirements.txt` first.")
    else:
        import uvicorn

        uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
