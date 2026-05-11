import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[2]
PANDOC_TOOL = ROOT / "tools" / "universal_file_generator_pandoc.py"
SPEC = importlib.util.spec_from_file_location("universal_file_generator_pandoc", PANDOC_TOOL)
assert SPEC is not None and SPEC.loader is not None
ufg = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ufg)


@pytest.mark.parametrize(
    ("data", "url"),
    [
        (
            {"asset.txt": "http://example.com/asset.txt"},
            "http://example.com/asset.txt",
        ),
        (
            [{"path": "asset.txt", "url": "http://example.com/asset.txt"}],
            "http://example.com/asset.txt",
        ),
    ],
)
def test_zip_url_download_non_200_is_error(monkeypatch, data, url):
    calls = []

    def fake_get(request_url, **kwargs):
        calls.append((request_url, kwargs))
        return SimpleNamespace(status_code=302, content=b"", text="Found")

    monkeypatch.setattr(ufg.requests, "get", fake_get)

    with pytest.raises(RuntimeError, match="HTTP 302"):
        ufg.FileGeneratorPandoc().generate_zip(data)

    assert calls == [(url, {"timeout": 10, "allow_redirects": False})]


@pytest.mark.parametrize(
    ("data", "url"),
    [
        (
            {"asset.txt": "http://example.com/asset.txt"},
            "http://example.com/asset.txt",
        ),
        (
            [{"path": "asset.txt", "url": "http://example.com/asset.txt"}],
            "http://example.com/asset.txt",
        ),
    ],
)
def test_zip_url_download_200_writes_entry(monkeypatch, data, url):
    calls = []

    def fake_get(request_url, **kwargs):
        calls.append((request_url, kwargs))
        return SimpleNamespace(status_code=200, content=b"downloaded")

    monkeypatch.setattr(ufg.requests, "get", fake_get)

    zip_bytes = ufg.FileGeneratorPandoc().generate_zip(data)

    assert calls == [(url, {"timeout": 10, "allow_redirects": False})]
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        assert zf.read("asset.txt") == b"downloaded"


def test_zip_url_download_honors_core_redirect_opt_in(monkeypatch):
    calls = []

    def fake_get(request_url, **kwargs):
        calls.append((request_url, kwargs))
        return SimpleNamespace(status_code=200, content=b"downloaded")

    monkeypatch.setenv("AIOHTTP_CLIENT_ALLOW_REDIRECTS", "True")
    monkeypatch.setattr(ufg.requests, "get", fake_get)

    ufg.FileGeneratorPandoc().generate_zip({"asset.txt": "http://example.com/asset.txt"})

    assert calls == [
        (
            "http://example.com/asset.txt",
            {"timeout": 10, "allow_redirects": True},
        )
    ]


def test_upload_honors_core_redirect_opt_in(monkeypatch):
    calls = []

    def fake_put(request_url, **kwargs):
        calls.append((request_url, kwargs))
        return SimpleNamespace(
            status_code=200,
            text="http://transfer-sh:8080/report.txt",
            headers={},
        )

    monkeypatch.setenv("AIOHTTP_CLIENT_ALLOW_REDIRECTS", "True")
    monkeypatch.setattr(ufg.requests, "put", fake_put)

    result = ufg._upload_file(b"content", "report.txt", "txt", len(b"content"))

    assert "File Generated and Uploaded Successfully" in result
    assert len(calls) == 1
    assert calls[0][1]["allow_redirects"] is True
