"""
Tests for FastAPI endpoints in app.py:
- Response shapes and types
- Pydantic models & JSON schemas (/openapi.json)
- Error handling (422 validation, 405 method not allowed, 404 not found)
- Greetings, retrieval, refusals, and edge cases
- Static file serving (/, /research, /static/...)
"""
import os
import sys
import pathlib
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

# Ensure fast start without loading dense model
os.environ["PLB_NO_DENSE"] = "1"

from fastapi.testclient import TestClient
import app

client = TestClient(app.app)


def test_health_endpoint_shape():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data["chunks"], int)
    assert data["chunks"] > 0
    assert isinstance(data["dense"], bool)
    assert isinstance(data["providers"], list)
    assert isinstance(data["corpus_verified"], bool)


def test_health_endpoint_method_not_allowed():
    resp = client.post("/api/health")
    assert resp.status_code == 405


def test_static_root_endpoint():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "PakLegalBench" in resp.text


def test_static_research_endpoint():
    resp = client.get("/research")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Research Dossier" in resp.text


def test_static_mounted_files():
    resp_banner = client.get("/static/banner.png")
    assert resp_banner.status_code == 200
    assert "image/png" in resp_banner.headers["content-type"]


def test_not_found_endpoint():
    resp = client.get("/nonexistent_path_404")
    assert resp.status_code == 404


def test_chat_greeting():
    for g in ["hello", "hi", "salam", "AoA", "help", "start"]:
        resp = client.post("/api/chat", json={"question": g})
        assert resp.status_code == 200
        data = resp.json()
        assert data["refused"] is False
        assert data["provider"] == "assistant"
        assert data["sources"] == []
        assert "PakLegalBench" in data["answer"]


def test_chat_validation_error_empty_body():
    resp = client.post("/api/chat", json={})
    assert resp.status_code == 422
    err = resp.json()
    assert "detail" in err


def test_chat_validation_error_wrong_types():
    resp = client.post("/api/chat", json={"question": "test", "history": "not-a-list"})
    assert resp.status_code == 422

    resp_bad_bool = client.post("/api/chat", json={"question": "test", "use_exact": "not-a-bool"})
    assert resp_bad_bool.status_code == 422


def test_chat_malformed_json():
    resp = client.post("/api/chat", content="invalid json {", headers={"content-type": "application/json"})
    assert resp.status_code == 422


def test_chat_method_not_allowed():
    resp = client.get("/api/chat")
    assert resp.status_code == 405


def test_chat_refusal_unknown_provision():
    resp = client.post("/api/chat", json={"question": "What is Section 999 PPC?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["refused"] is True
    assert data["provider"] is None
    assert "could not find" in data["answer"]


def test_chat_refusal_out_of_scope():
    resp = client.post("/api/chat", json={"question": "What is the corporate tax rate for a private limited company?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["refused"] is True
    assert data["provider"] is None


def test_chat_empty_and_whitespace_query():
    for q in ["", "   ", " \t\n\r "]:
        resp = client.post("/api/chat", json={"question": q})
        assert resp.status_code == 200
        data = resp.json()
        assert data["refused"] is True
        assert data["provider"] is None
        assert data["sources"] == []


def test_chat_special_symbols_query():
    resp = client.post("/api/chat", json={"question": "!@#$%^&*()_+=-"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["refused"] is True
    assert data["provider"] is None
    assert data["sources"] == []


def test_chat_valid_response_shape_with_mocked_llm():
    with patch("llm.answer", return_value=("Under Section 497 CrPC, bail may be granted.", "mock-groq")):
        resp = client.post("/api/chat", json={"question": "u/s 497 CrPC"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["refused"] is False
        assert data["provider"] == "mock-groq"
        assert len(data["sources"]) > 0
        src = data["sources"][0]
        assert "citation" in src and isinstance(src["citation"], str)
        assert "marginal_note" in src and isinstance(src["marginal_note"], str)
        assert "act" in src and isinstance(src["act"], str)
        assert "text" in src and isinstance(src["text"], str)
        assert "score" in src and isinstance(src["score"], float)
        assert "channels" in src and isinstance(src["channels"], list)
        assert "verified" in src and isinstance(src["verified"], bool)
        assert "source_url" in src and isinstance(src["source_url"], str)
        assert "cross_references" in src and isinstance(src["cross_references"], list)


def test_chat_multi_citation_query():
    with patch("llm.answer", return_value=("302 PPC covers murder while 497 CrPC covers bail.", "mock-groq")):
        resp = client.post("/api/chat", json={"question": "302 PPC and 497 CrPC"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["refused"] is False
        assert len(data["sources"]) >= 2
        citations = [s["citation"] for s in data["sources"][:2]]
        assert any("302" in c for c in citations)
        assert any("497" in c for c in citations)


def test_chat_urdu_query():
    with patch("llm.answer", return_value=("Under PPC Section 302, qatl-i-amd is defined and punished.", "mock-groq")):
        resp = client.post("/api/chat", json={"question": "دفعہ ۳۰۲ تعزیرات پاکستان"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["refused"] is False
        assert len(data["sources"]) >= 1
        assert "302" in data["sources"][0]["citation"]


def test_openapi_schema_definitions():
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    schema = resp.json()
    schemas = schema["components"]["schemas"]
    assert "ChatRequest" in schemas
    assert "ChatResponse" in schemas
    assert "HealthResponse" in schemas
    assert "SourceItem" in schemas
    # Verify paths are registered
    paths = schema["paths"]
    assert "/api/chat" in paths
    assert "/api/health" in paths
    assert "/api/benchmarks" in paths
    assert "/api/graph" in paths
    assert "/" in paths
    assert "/research" in paths


def test_benchmarks_endpoint():
    resp = client.get("/api/benchmarks")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, dict)
    for expected_key in ["retrieval", "law_gat", "stanford", "redteam"]:
        assert expected_key in data


def test_graph_endpoint():
    # Summary
    resp = client.get("/api/graph")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_provisions_with_edges"] > 0
    assert data["total_directed_edges"] > 0
    assert isinstance(data["sample_nodes"], list)

    # Specific provision query
    resp_prov = client.get("/api/graph?provision=PPC:302")
    assert resp_prov.status_code == 200
    data_prov = resp_prov.json()
    assert data_prov["provision"] == "PPC:302"
    assert isinstance(data_prov["outbound_citations"], list)
    assert isinstance(data_prov["inbound_citations"], list)
