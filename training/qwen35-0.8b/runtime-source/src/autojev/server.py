from __future__ import annotations

import base64
import binascii
import hmac
import os
import threading
import time
import uuid
from _thread import LockType
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, cast

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, Response
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator
from starlette.concurrency import run_in_threadpool
from starlette.middleware.base import RequestResponseEndpoint

from autojev.types import Answer, DecisionInput, DecisionResponse, JSONValue
from autojev.types import Question as DecisionQuestion

if TYPE_CHECKING:
    from autojev.model import DecisionModel

type Content = str | dict[str, JsonValue] | list[JsonValue]
DEFAULT_MODEL = "autojev-qwen3.8-27b"
ALIASES = {"autojev", "jev-latest", "jev-preview", "jev-1.13.0", DEFAULT_MODEL}


@dataclass
class Service:
    model: DecisionModel | None = None
    name: str = DEFAULT_MODEL
    checkpoint: str = "checkpoints/selected"
    release_date: str = ""
    lock: LockType = field(default_factory=threading.Lock)


service = Service()


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instructions: Content | None = None


class Choice(Question):
    type: Literal["choice"]
    criteria: dict[str, Content | None] = Field(min_length=1, max_length=255)


class Score(Question):
    type: Literal["score"]
    criteria: list[Content] = Field(min_length=2, max_length=10)


class Noul(Question):
    type: Literal["noul"]
    criteria: dict[Literal["true", "false"], Content | None] | None = None


class EvaluationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    state: Content
    questions: dict[str, Annotated[Choice | Score | Noul, Field(discriminator="type")]] = Field(min_length=1)
    images: list[str] = Field(default_factory=list, max_length=4)

    @field_validator("model")
    @classmethod
    def known_model(cls, value: str) -> str:
        if value not in ALIASES | {service.name}:
            raise ValueError(f"Unknown model. Use {service.name} or jev-latest.")
        return value

    @field_validator("images")
    @classmethod
    def valid_images(cls, values: list[str]) -> list[str]:
        for value in values:
            if len(value) > 12_000_000:
                raise ValueError("Each image must be at most 8 MB before base64 encoding.")
            header, separator, encoded = value.partition(",")
            if not separator or header not in {
                "data:image/png;base64",
                "data:image/jpeg;base64",
                "data:image/webp;base64",
            }:
                raise ValueError("Images must be base64 PNG, JPEG, or WebP data URLs.")
            try:
                content = base64.b64decode(encoded, validate=True)
                if len(content) > 8_000_000:
                    raise ValueError("Each image must be at most 8 MB.")
                with Image.open(BytesIO(content)) as image:
                    if image.width * image.height > 16_000_000:
                        raise ValueError("Each image must have at most 16 million pixels.")
                    if image.format not in {"PNG", "JPEG", "WEBP"}:
                        raise ValueError("Unsupported image format.")
                    image.verify()
            except (
                binascii.Error,
                OSError,
                SyntaxError,
                UnidentifiedImageError,
                Image.DecompressionBombError,
            ) as error:
                raise ValueError("Invalid image data.") from error
        return values


def authenticate(authorization: str | None = Header(default=None)) -> None:
    key = os.getenv("AUTOJEV_API_KEY")
    if key and not hmac.compare_digest((authorization or "").encode(), f"Bearer {key}".encode()):
        raise HTTPException(401, "Missing or invalid API key.", headers={"WWW-Authenticate": "Bearer"})


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    from autojev.model import DecisionModel

    service.checkpoint = os.getenv("AUTOJEV_CHECKPOINT", "checkpoints/selected")
    service.model = await run_in_threadpool(DecisionModel, checkpoint=service.checkpoint)
    service.name = f"autojev-{service.model.base_model.rsplit('/', 1)[-1].lower()}"
    modified = (Path(service.checkpoint) / "decision_config.json").stat().st_mtime
    service.release_date = datetime.fromtimestamp(modified, UTC).date().isoformat()
    try:
        yield
    finally:
        service.model = None


app = FastAPI(title="AutoJev", version="0.2.0", lifespan=lifespan)


@app.middleware("http")
async def request_metadata(request: Request, call_next: RequestResponseEndpoint) -> Response:
    started, identifier = time.perf_counter(), uuid.uuid4().hex
    response = await call_next(request)
    response.headers["x-typesafe-request-id"] = identifier
    response.headers["x-request-id"] = identifier
    response.headers["server-timing"] = f"total;dur={(time.perf_counter() - started) * 1000:.1f}"
    return response


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, error: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={"detail": [{"loc": item["loc"], "msg": item["msg"], "type": item["type"]} for item in error.errors()]},
    )


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def playground() -> str:
    return Path(__file__).with_name("playground.html").read_text()


@app.get("/health", response_model=None)
def health() -> dict[str, JSONValue]:
    return {
        "status": "ready" if service.model is not None else "loading",
        "model": service.name,
        "checkpoint": service.checkpoint,
        "authentication": bool(os.getenv("AUTOJEV_API_KEY")),
        "modalities": ["text", "image"],
    }


@app.get("/v1/models", dependencies=[Depends(authenticate)], response_model=None)
def models() -> dict[str, JSONValue]:
    return {
        "models": [
            {
                "name": name,
                "description": "Local AutoJev text and image decisions.",
                "release_date": service.release_date,
            }
            for name in sorted(ALIASES | {service.name})
        ]
    }


def predict(model: DecisionModel, body: EvaluationRequest) -> DecisionResponse:
    import torch

    from autojev.model import answer

    questions = {
        key: cast(DecisionQuestion, question.model_dump(exclude_none=True)) for key, question in body.questions.items()
    }
    identifiers = list(questions)
    rows: list[DecisionInput] = [
        {"state": body.state, "question": question, "images": list(body.images)} for question in questions.values()
    ]
    answers: dict[str, Answer] = {}
    input_tokens = 0
    with torch.inference_mode():
        for start in range(0, len(rows), 8):
            batch = model.prepare(rows[start : start + 8])
            distributions: list[list[float]] = (model(batch) / model.temperature).softmax(-1).cpu().tolist()
            for identifier, values, count in zip(
                identifiers[start : start + 8], distributions, batch.counts, strict=True
            ):
                answers[identifier] = answer(questions[identifier], values[:count])
            input_tokens += batch.input_tokens
    return {"model": service.name, "answers": answers, "usage": {"input_tokens": input_tokens, "output_tokens": 0}}


@app.post("/v1/systemone", dependencies=[Depends(authenticate)], response_model=None)
async def system_one(body: EvaluationRequest) -> DecisionResponse:
    model = service.model
    if model is None:
        raise HTTPException(503, "The model is not ready.")
    if not service.lock.acquire(blocking=False):
        raise HTTPException(529, "The model is busy. Retry shortly.", headers={"Retry-After": "1"})
    try:
        return await run_in_threadpool(predict, model, body)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    finally:
        service.lock.release()


def main() -> None:
    import uvicorn

    uvicorn.run("autojev.server:app", host=os.getenv("AUTOJEV_HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")))


if __name__ == "__main__":
    main()
