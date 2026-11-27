"""Validate a live authenticated DRISHTI NVR from a Windows or Linux terminal."""

import argparse
import base64
import getpass
import hashlib
import http.client
import http.cookiejar
import json
import os
import secrets
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import quote, urlencode, urljoin, urlsplit


class Client:
    def __init__(self, url):
        self.base = url.rstrip("/") + "/"
        self.parts = urlsplit(self.base)
        if (
            self.parts.hostname not in {"localhost", "127.0.0.1", "::1"}
            and self.parts.scheme != "https"
        ):
            raise ValueError("Choose HTTPS or a local loopback address")
        self.jar = http.cookiejar.CookieJar()
        self.http = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(self.jar)
        )

    def read(self, path, data=None, headers=None, timeout=120):
        req = urllib.request.Request(
            urljoin(self.base, path), data, headers=headers or {}
        )
        with self.http.open(req, timeout=timeout) as response:
            return response.read()

    def login(self, password):
        self.read(
            "api/login",
            json.dumps({"user": "admin", "password": password}).encode(),
            {"Content-Type": "application/json"},
        )

    def json(self, path):
        return json.loads(self.read(path))

    def websocket(self):
        p = self.parts
        conn_type = (
            http.client.HTTPSConnection
            if p.scheme == "https"
            else http.client.HTTPConnection
        )
        conn = conn_type(p.hostname, p.port, timeout=40)
        key = base64.b64encode(secrets.token_bytes(16)).decode()
        req = urllib.request.Request(self.base)
        self.jar.add_cookie_header(req)
        try:
            conn.request(
                "GET",
                p.path.rstrip("/") + "/ws",
                headers={
                    "Upgrade": "websocket",
                    "Connection": "Upgrade",
                    "Sec-WebSocket-Version": "13",
                    "Sec-WebSocket-Key": key,
                    "Cookie": req.get_header("Cookie") or "",
                    "Origin": f"{p.scheme}://{p.netloc}",
                },
            )
            res = conn.getresponse()
            accept = base64.b64encode(
                hashlib.sha1(
                    (key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()
                ).digest()
            ).decode()
            if res.status != 101 or res.getheader("Sec-WebSocket-Accept") != accept:
                raise RuntimeError(f"WebSocket upgrade returned {res.status}")
            return res.status
        finally:
            conn.close()

    def upload(self, data, *, background=False):
        boundary = secrets.token_hex(24)
        body = (
            (
                f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="clip.mp4"\r\n'
                "Content-Type: video/mp4\r\n\r\n"
            ).encode()
            + data
            + f"\r\n--{boundary}--\r\n".encode()
        )
        return json.loads(
            self.read(
                "api/drishti/analysis" + ("?background=true" if background else ""),
                body,
                {
                    "Content-Type": "multipart/form-data; boundary=" + boundary,
                    "X-CSRF-TOKEN": "1",
                },
                timeout=360,
            )
        )


def verify_camera(client, camera, wait_seconds=180):
    """Wait for the named camera to yield real event, review, and recording evidence."""
    if camera not in client.json("api/config").get("cameras", {}):
        raise ValueError("Unknown camera: " + camera)
    encoded = quote(camera, safe="")
    query = urlencode({"cameras": camera, "limit": 30})
    deadline = time.monotonic() + wait_seconds
    evidence = {"camera": camera}
    while time.monotonic() < deadline:
        events = client.json("api/events?" + query)
        reviews = client.json("api/review?" + query)
        segments = client.json(f"api/{encoded}/recordings")
        evidence.update(event_rows=len(events), review_rows=len(reviews), recording_segments=len(segments))
        if "snapshot_jpeg_bytes" not in evidence:
            for event in events:
                if event.get("camera") != camera or not event.get("has_snapshot"):
                    continue
                try:
                    snapshot = client.read("api/events/" + quote(event["id"], safe="") + "/snapshot.jpg")
                except urllib.error.HTTPError as exc:
                    if exc.code == 404:
                        continue
                    raise
                if not snapshot.startswith(b"\xff\xd8"):
                    raise RuntimeError("The camera snapshot did not contain a JPEG")
                evidence.update(snapshot_jpeg_bytes=len(snapshot), snapshot_label=event["label"])
                break
        if segments and "recorded_clip_bytes" not in evidence:
            segment = next((r for r in reversed(segments) if r["end_time"] - r["start_time"] >= 4), None)
            if segment:
                start = int(segment["start_time"]) + 1
                clip = client.read(f"api/{encoded}/start/{start}/end/{start + 2}/clip.mp4")
                if clip[4:8] != b"ftyp":
                    raise RuntimeError("The recording download did not contain an MP4")
                evidence["recorded_clip_bytes"] = len(clip)
        if reviews and "snapshot_jpeg_bytes" in evidence and "recorded_clip_bytes" in evidence:
            return evidence
        time.sleep(3)
    raise RuntimeError("Camera evidence incomplete: " + json.dumps(evidence))


def verify(url, password, video=None, camera=None):
    c = Client(url)
    checks = {}
    try:
        c.read("api/config")
    except urllib.error.HTTPError as exc:
        if exc.code not in (401, 403):
            raise
        checks["unauthenticated_config"] = exc.code
    else:
        raise RuntimeError(
            "Config accessible before login. Use the authenticated port."
        )
    c.login(password)
    for path in (
        "config",
        "profile",
        "profiles",
        "notices",
        "stats",
        "notices/muted_checks",
    ):
        data = c.json("api/" + path)
        if not isinstance(data, (dict, list)):
            raise ValueError("Unexpected response: " + path)
        if path == "config" and (not data.get("cameras") or data.get("safe_mode")):
            raise RuntimeError("Camera config missing or NVR in safe mode")
        checks["/api/" + path] = 200
    checks["websocket_upgrade"] = c.websocket()
    manifest = c.json("site.webmanifest")
    for icon in manifest["icons"]:
        if "BASE_PATH" in icon["src"] or not c.read(icon["src"]).startswith(
            b"\x89PNG\r\n\x1a\n"
        ):
            raise RuntimeError("Broken application icon")
    checks["manifest_icons"] = len(manifest["icons"])
    for lang in ("en", "hi", "te"):
        if not c.json(f"locales/{lang}/views/exports.json")["analysis"]["run"]:
            raise RuntimeError("Missing language: " + lang)
    checks["translation_files"] = ["en", "hi", "te"]
    if video:
        try:
            c.upload(b"invalid media")
        except urllib.error.HTTPError as exc:
            if exc.code != 422:
                raise
            checks["invalid_upload"] = 422
        else:
            raise RuntimeError("Invalid media was accepted")
        clip = video.read_bytes()
        result = c.upload(clip, background=True)
        if result.get("status") != "running":
            raise RuntimeError("The NVR did not acknowledge the background job")
        deadline = time.monotonic() + 1230
        while time.monotonic() < deadline:
            rows = c.json("api/drishti/analysis")
            if any(row["id"] == result["id"] for row in rows):
                break
            state = c.json("api/drishti/analysis/status")
            if state.get("id") == result["id"] and state["status"] in {
                "invalid",
                "failed",
                "interrupted",
            }:
                raise RuntimeError("NVR analysis ended with status " + state["status"])
            time.sleep(2)
        else:
            raise RuntimeError("Timed out waiting for the NVR analysis")
        prefix = "api/drishti/analysis/" + result["id"] + "/"
        r = c.json(prefix + "report.json")
        movie = c.read(prefix + "review.mp4")
        if (
            r["input_sha256"] != hashlib.sha256(clip).hexdigest()
            or len(r["samples"]) != r["sample_count"]
            or movie[4:8] != b"ftyp"
            or result["id"] not in {a["id"] for a in c.json("api/drishti/analysis")}
        ):
            raise RuntimeError("Persisted analysis verification failed")
        checks["upload"] = {
            k: r[k] for k in ("model", "provider", "sample_count", "box_count")
        }
        checks["downloaded_review_bytes"] = len(movie)
    if camera:
        checks["camera_evidence"] = verify_camera(c, camera)
    return checks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8971")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--camera", help="Also verify a real event snapshot, review and recording for this camera")
    args = parser.parse_args()
    print(
        json.dumps(
            verify(
                args.url,
                os.environ.get("DRISHTI_TEST_PASSWORD")
                or getpass.getpass("Admin password: "),
                args.video,
                args.camera,
            ),
            indent=2,
        )
    )
