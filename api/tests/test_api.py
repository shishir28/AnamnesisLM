from dataclasses import replace
from threading import Event, Thread

import pytest
from fastapi.testclient import TestClient

from anamnesis_api.application.extraction import ExtractionService, ModelBusyError
from anamnesis_api.config import Settings
from anamnesis_api.domain.extraction import ModelOutput
from anamnesis_api.presentation.http import create_app

NOTE = "Patient denies chest pain."
FACT = {
    "kind": "symptom",
    "mention": "chest pain",
    "assertion": "negated",
    "experiencer": "patient",
    "temporality": "unspecified",
    "evidence": NOTE,
    "evidence_start": 0,
    "evidence_end": len(NOTE),
    "dose": None,
    "unit": None,
    "route": None,
    "frequency": None,
}


class FakeBackend:
    def __init__(self):
        self.output = ModelOutput(
            '{"facts": []}', [dict(FACT)], "test@step1", generation_token_limit=100
        )
        self.calls = []

    def info(self):
        return {"model_version": "test@step1", "context_length": 512, "device": "cpu"}

    def extract(self, note, max_new_tokens):
        self.calls.append((note, max_new_tokens))
        return self.output


@pytest.fixture
def backend():
    return FakeBackend()


@pytest.fixture
def client(backend):
    with TestClient(create_app(Settings(), backend)) as instance:
        yield instance


def test_health_and_ready(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").status_code == 200
    assert client.get("/model").json()["model_version"] == "test@step1"


def test_extract_valid_facts(client, backend):
    response = client.post("/extract", json={"note": NOTE})
    assert response.status_code == 200
    body = response.json()
    assert body["valid"] is True
    assert body["facts"] == [FACT]
    assert body["validation_errors"] == []
    assert body["elapsed_ms"] >= 0
    assert backend.calls == [(NOTE, 400)]


@pytest.mark.parametrize("note", ["", "   ", "x" * 12001, None, 12])
def test_invalid_request_does_not_call_model(client, backend, note):
    assert client.post("/extract", json={"note": note}).status_code == 422
    assert backend.calls == []


@pytest.mark.parametrize("budget", [0, 1025, 1.5, "10", True])
def test_invalid_generation_budget(client, backend, budget):
    assert client.post("/extract", json={"note": NOTE, "max_new_tokens": budget}).status_code == 422
    assert backend.calls == []


def test_validation_error_does_not_echo_note(client):
    response = client.post("/extract", json={"note": "private synthetic marker", "extra": "x"})
    assert response.status_code == 422
    assert "private synthetic marker" not in response.text


def test_preserves_note_whitespace(client, backend):
    client.post("/extract", json={"note": "  Synthetic note.\n"})
    assert backend.calls[0][0] == "  Synthetic note.\n"


def test_invalid_json_is_returned_not_repaired(client, backend):
    backend.output = ModelOutput("{broken", None, "test", error="Invalid JSON")
    response = client.post("/extract", json={"note": NOTE})
    body = response.json()
    assert response.status_code == 200
    assert body["valid"] is False
    assert body["raw_output"] == "{broken"
    assert body["facts"] is None
    assert body["validation_errors"] == ["Invalid JSON"]


@pytest.mark.parametrize(
    "change",
    [
        {"evidence_end": 999},
        {"evidence": "Invented evidence"},
        {"assertion": "maybe"},
        {"evidence_start": True},
        {"mention": "unmentioned disease"},
        {"dose": "10"},
    ],
)
def test_invalid_fact_remains_visible(client, backend, change):
    fact = {**FACT, **change}
    backend.output = replace(backend.output, facts=[fact])
    body = client.post("/extract", json={"note": NOTE}).json()
    assert body["valid"] is False
    assert body["facts"] == [fact]
    assert body["validation_errors"]


def test_empty_facts_are_structurally_valid(client, backend):
    backend.output = replace(backend.output, facts=[])
    assert client.post("/extract", json={"note": NOTE}).json()["valid"] is True


def test_nonfinite_generated_values_preserve_raw_output(client, backend):
    backend.output = replace(
        backend.output, raw_output='{"facts":[{"dose":NaN}]}', facts=[{"dose": float("nan")}]
    )
    response = client.post("/extract", json={"note": NOTE})
    assert response.status_code == 200
    assert response.json()["valid"] is False
    assert response.json()["facts"] is None
    assert "NaN" in response.json()["raw_output"]


def test_server_caps_generation_budget(client, backend):
    body = client.post("/extract", json={"note": NOTE, "max_new_tokens": 1024}).json()
    assert backend.calls[0][1] == 400
    assert body["warnings"]


def test_missing_checkpoint_keeps_health_but_not_readiness(tmp_path):
    settings = Settings(checkpoint_path=tmp_path / "missing.pt")
    with TestClient(create_app(settings)) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503
        assert client.get("/model").json()["ready"] is False
        assert client.post("/extract", json={"note": NOTE}).status_code == 503


def test_inference_error_does_not_leak_exception(client, backend):
    def broken(*args):
        raise RuntimeError("private failure detail")

    backend.extract = broken
    response = client.post("/extract", json={"note": NOTE})
    assert response.status_code == 500
    assert "private failure detail" not in response.text


def test_busy_model_returns_429(client):
    # Hold the same service lock used by this request.
    service = client.app.state.service
    service._lock.acquire()
    try:
        assert client.post("/extract", json={"note": NOTE}).status_code == 429
    finally:
        service._lock.release()


def test_service_rejects_concurrent_generation_and_releases_lock(backend):
    entered, release = Event(), Event()

    def blocking(note, budget):
        entered.set()
        release.wait(timeout=5)
        return backend.output

    backend.extract = blocking
    service = ExtractionService(backend)
    worker = Thread(target=service.extract, args=(NOTE, 10))
    worker.start()
    try:
        assert entered.wait(timeout=5)
        with pytest.raises(ModelBusyError):
            service.extract(NOTE, 10)
    finally:
        release.set()
        worker.join(timeout=5)
    assert service.extract(NOTE, 10) == backend.output
