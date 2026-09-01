import io
import json

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app import storage
from app.models import Base
from tests.conftest import make_video


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(storage.settings, "media_dir", str(tmp_path / "media"))
    engine = create_engine(f"sqlite:///{tmp_path}/t.db")
    Base.metadata.create_all(engine)
    factory = sessionmaker(engine, expire_on_commit=False)
    import app.api as api

    monkeypatch.setattr(api, "SessionLocal", factory)
    monkeypatch.setattr(api, "get_vlm_client", lambda: __import__(
        "app.vlm.client", fromlist=["FakeVlmClient"]
    ).FakeVlmClient())
    from app.main import app

    return TestClient(app)


def test_upload_analyze_report_flow(client, tmp_path):
    path = make_video(tmp_path / "c.mp4", seconds=12, moving=True)
    with open(path, "rb") as f:
        r = client.post(
            "/api/videos/upload",
            files={"file": ("c.mp4", f, "video/mp4")},
            data={"project_name": "ЖК Тест"},
        )
    assert r.status_code == 200
    vid = r.json()["video_id"]

    r = client.post(f"/api/videos/{vid}/analyze")
    assert r.status_code == 202

    r = client.get(f"/api/videos/{vid}/report")
    assert r.status_code == 200
    report = r.json()
    assert report["video_id"] == vid and report["findings"]

    frame_id = report["findings"][0]["evidence"][0]["frame_id"]
    r = client.get(f"/api/frames/{frame_id}?thumb=1")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"

    r = client.get("/api/videos")
    assert r.json()[0]["id"] == vid and r.json()[0]["status"] == "done"


def test_report_before_done_returns_202(client, tmp_path):
    path = make_video(tmp_path / "c2.mp4", seconds=6)
    with open(path, "rb") as f:
        r = client.post("/api/videos/upload", files={"file": ("c2.mp4", f, "video/mp4")})
    vid = r.json()["video_id"]
    assert client.get(f"/api/videos/{vid}/report").status_code == 202
    assert client.get("/api/videos/vid_missing/report").status_code == 404


def test_upload_sanitizes_filename(client):
    r = client.post(
        "/api/videos/upload",
        files={"file": ("../../evil.mp4", io.BytesIO(b"x"), "video/mp4")},
    )
    vid = r.json()["video_id"]
    listed = next(v for v in client.get("/api/videos").json() if v["id"] == vid)
    assert listed["filename"] == "evil.mp4"


def test_sse_status_stream_ends_on_done(client, tmp_path):
    path = make_video(tmp_path / "c3.mp4", seconds=6)
    with open(path, "rb") as f:
        vid = client.post(
            "/api/videos/upload", files={"file": ("c3.mp4", f, "video/mp4")}
        ).json()["video_id"]
    client.post(f"/api/videos/{vid}/analyze")
    with client.stream("GET", f"/api/videos/{vid}/status") as r:
        events = [
            json.loads(line[6:]) for line in r.iter_lines() if line.startswith("data: ")
        ]
    assert events[-1]["status"] == "done"
