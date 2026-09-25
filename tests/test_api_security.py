"""
Automated Security Tests for Pathology Active Learning API.

Evaluates endpoints against OWASP Top 10 vulnerabilities:
- Unrestricted File Upload & MIME filtering
- Payload Size Limits (DoS prevention)
- Decompression Bomb defenses
- Path traversal prevention
- Security Headers (X-Content-Type-Options, X-Frame-Options, CSP)
"""

import io
import pytest
from fastapi.testclient import TestClient

from app import app
from config import settings


@pytest.fixture
def client():
    return TestClient(app)


def test_security_headers(client):
    """Verify essential security headers are returned on responses."""
    response = client.get("/api/config")
    assert response.status_code == 200
    assert response.headers.get("x-content-type-options") == "nosniff"
    assert response.headers.get("x-frame-options") == "DENY"
    assert "Content-Security-Policy" in response.headers or "content-security-policy" in response.headers


def test_upload_invalid_mime_type(client):
    """Ensure non-image and unallowed MIME types are rejected with 415."""
    fake_exe = io.BytesIO(b"MZ\x90\x00\x03\x00\x00\x00")
    response = client.post(
        "/api/test/upload",
        files={"file": ("malicious.exe", fake_exe, "application/x-msdownload")},
    )
    assert response.status_code == 415
    assert "Unsupported image type" in response.json()["detail"]


def test_upload_oversized_payload(client):
    """Ensure uploads exceeding max_upload_bytes are rejected with 413."""
    oversized = io.BytesIO(b"A" * (settings.max_upload_bytes + 1024))
    response = client.post(
        "/api/test/upload",
        files={"file": ("huge.jpg", oversized, "image/jpeg")},
    )
    assert response.status_code == 413
    assert "upload size limit" in response.json()["detail"].lower()


def test_upload_corrupt_image(client):
    """Ensure malformed or unreadable images return 400 Bad Request."""
    corrupt_image = io.BytesIO(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x01\x00\x00\x00\x01\x00\x00")
    response = client.post(
        "/api/test/upload",
        files={"file": ("bomb.png", corrupt_image, "image/png")},
    )
    assert response.status_code == 400


def test_sandbox_save_path_traversal_blocked(client):
    """Ensure path traversal payloads in sandbox_filename are rejected by regex pattern or path validation."""
    response = client.post(
        "/api/test/save",
        json={
            "sandbox_filename": "../../../etc/passwd",
            "annotations": [
                {
                    "class_label": "Benign",
                    "x_center": 0.5,
                    "y_center": 0.5,
                    "width": 0.2,
                    "height": 0.2,
                    "confidence": 1.0,
                }
            ],
        },
    )
    assert response.status_code in [400, 422]


def test_save_request_max_annotations_limit(client):
    """Ensure oversized annotations list exceeding max_length is rejected by validation."""
    excessive_boxes = [
        {
            "class_label": "Benign",
            "x_center": 0.5,
            "y_center": 0.5,
            "width": 0.1,
            "height": 0.1,
            "confidence": 1.0,
        }
        for _ in range(5001)
    ]
    response = client.post(
        "/api/tile/1/save",
        json={"annotations": excessive_boxes},
    )
    assert response.status_code == 422  # Pydantic validation error for max_length exceeded
