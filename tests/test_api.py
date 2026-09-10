"""
tests/test_api.py
-----------------
Unit tests for the FastAPI REST API and frontend serving endpoints.
"""

import pytest
from fastapi.testclient import TestClient

from api.app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_check_json(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "CNN Orthophoto" in data["service"]


def test_root_json_for_non_browser(client):
    res = client.get("/", headers={"Accept": "application/json"})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"


def test_root_serves_html_for_browser(client):
    res = client.get("/", headers={"Accept": "text/html,application/xhtml+xml"})
    assert res.status_code == 200
    assert "text/html" in res.headers.get("content-type", "")
    assert "<!DOCTYPE html>" in res.text or "OrthoSeg" in res.text


def test_static_assets(client):
    res_css = client.get("/style.css")
    assert res_css.status_code == 200
    assert "text/css" in res_css.headers.get("content-type", "")

    res_js = client.get("/app.js")
    assert res_js.status_code == 200
    assert "javascript" in res_js.headers.get("content-type", "")


def test_classes_endpoint(client):
    res = client.get("/classes")
    assert res.status_code == 200
    data = res.json()
    assert "classes" in data
    assert "building" in data["classes"]
    assert "road" in data["classes"]
    assert "vegetation" in data["classes"]


def test_model_info_endpoint(client):
    res = client.get("/model/info")
    assert res.status_code == 200
    data = res.json()
    assert "architecture" in data
    assert "encoder" in data
    assert data["num_classes"] == 4


def test_folders_scan_nonexistent(client):
    res = client.get("/folders/scan?folder_path=nonexistent_dir_12345")
    assert res.status_code == 404
    assert "does not exist" in res.json()["detail"]


def test_folders_scan_valid_folder(client, tmp_path):
    # Create fake GeoTIFF primary and mask
    f1 = tmp_path / "area1.tif"
    f1.write_bytes(b"dummy geotiff data")
    f1_mask = tmp_path / "area1_mask.tif"
    f1_mask.write_bytes(b"dummy mask data")
    f2 = tmp_path / "area2.tif"
    f2.write_bytes(b"dummy geotiff 2")

    res = client.get(f"/folders/scan?folder_path={str(tmp_path)}")
    assert res.status_code == 200
    data = res.json()
    assert data["total_files"] == 3
    assert data["primary_count"] == 2
    assert data["mask_count"] == 1
    names = [f["name"] for f in data["primary_files"]]
    assert "area1.tif" in names
    assert "area2.tif" in names

    # area1 has paired mask
    area1_info = next(f for f in data["primary_files"] if f["name"] == "area1.tif")
    assert area1_info["has_paired_mask"] is True

    # area2 does not have paired mask
    area2_info = next(f for f in data["primary_files"] if f["name"] == "area2.tif")
    assert area2_info["has_paired_mask"] is False


def test_predict_folder_nonexistent(client):
    res = client.post("/predict/folder?folder_path=nonexistent_dir_99999")
    assert res.status_code == 404
    assert "does not exist" in res.json()["detail"]
