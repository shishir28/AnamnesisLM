import logging
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from anamnesis_api.application.extraction import (
    ExtractionBackend,
    ExtractionService,
    InvalidNoteError,
    ModelBusyError,
)
from anamnesis_api.config import Settings
from anamnesis_api.domain.extraction import validate_facts
from anamnesis_api.infrastructure.model import LocalModelBackend

logger = logging.getLogger(__name__)


class ExtractionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    note: str = Field(min_length=1, max_length=12000)
    max_new_tokens: int = Field(default=400, ge=1, le=1024)

    @field_validator("note")
    @classmethod
    def reject_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Note must not be blank.")
        return value  # Preserve original whitespace for evidence offsets.


class ExtractionResponse(BaseModel):
    raw_output: str
    facts: list[dict[str, Any]] | None
    model_version: str
    valid: bool
    validation_errors: list[str]
    warnings: list[str]
    elapsed_ms: float
    generation_token_limit: int


def create_app(
    settings: Settings | None = None, backend: ExtractionBackend | None = None
) -> FastAPI:
    config = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = None
        app.state.model_error = "Model is not loaded."
        try:
            loaded = backend or LocalModelBackend.load(config.checkpoint_path, config.device)
            app.state.service = ExtractionService(loaded)
            app.state.model_error = None
        except Exception:
            # Do not log exception text: it could include paths or model input.
            logger.error("Model startup failed. Check checkpoint configuration and device support.")
            app.state.model_error = (
                "Model unavailable. Check the checkpoint and device configuration."
            )
        yield
        app.state.service = None

    app = FastAPI(title="AnamnesisLM", version="0.1.0", lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError):
        # FastAPI's default errors include rejected inputs. Do not echo note text.
        return JSONResponse(
            status_code=422,
            content={
                "detail": [
                    {"loc": list(error["loc"]), "msg": error["msg"], "type": error["type"]}
                    for error in exc.errors()
                ]
            },
        )

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/ready")
    def ready(request: Request):
        if request.app.state.service is None:
            raise HTTPException(503, request.app.state.model_error)
        return {"status": "ready"}

    @app.get("/model")
    def model_info(request: Request):
        service = request.app.state.service
        if service is None:
            return {"ready": False, "error": request.app.state.model_error}
        return {"ready": True, "max_new_tokens": config.max_new_tokens, **service.backend.info()}

    @app.post("/extract", response_model=ExtractionResponse)
    def extract(payload: ExtractionRequest, request: Request):
        service = request.app.state.service
        if service is None:
            raise HTTPException(503, request.app.state.model_error)
        start = time.perf_counter()
        try:
            output = service.extract(
                payload.note, min(payload.max_new_tokens, config.max_new_tokens)
            )
        except InvalidNoteError as exc:
            raise HTTPException(422, str(exc)) from exc
        except ModelBusyError as exc:
            raise HTTPException(
                429, "Model is busy. Retry after the current request completes."
            ) from exc
        except Exception as exc:
            logger.error("Model inference failed.")
            raise HTTPException(500, "Inference failed. Check server configuration.") from exc
        errors = validate_facts(payload.note, output)
        warnings = list(output.warnings)
        if payload.max_new_tokens > config.max_new_tokens:
            warnings.append(f"Server generation limit is {config.max_new_tokens} tokens.")
        return ExtractionResponse(
            raw_output=output.raw_output,
            facts=output.facts,
            model_version=output.model_version,
            valid=not errors,
            validation_errors=errors,
            warnings=warnings,
            elapsed_ms=round((time.perf_counter() - start) * 1000, 2),
            generation_token_limit=output.generation_token_limit,
        )

    return app
