"""Minimal async render endpoint for the app's PipelineRenderer.

    python3 -m dreamrender.serve --port 8787 --out out/jobs

POST /render        body: dream spec v1 JSON (or {"text": "..."})  -> 202 {"job": id, "status_url": "/jobs/<id>"}
GET  /jobs/<id>     -> {"status": "queued|running|done|failed", "result": {"kind": "video", "src": "/files/<id>/clip.mp4", "poster": ...}}
GET  /files/...     -> rendered files
One render at a time (CPU bound); CORS open for local testing.
"""
from __future__ import annotations
import argparse, json, os, queue, threading, traceback, uuid
from dataclasses import asdict
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from .parse import parse
from .stylize import look_from_spec
from .app_spec import from_app_spec_multi
from .pipeline import run, run_multi, run_app_spec, ContentDeclined

JOBS: dict = {}
Q: queue.Queue = queue.Queue()


def worker(out_root, frames, samples):
    while True:
        jid, body = Q.get()
        JOBS[jid]["status"] = "running"
        try:
            if not ("look" in body or "scenes" in body):      # plain text: wrap it so it goes through the same gate
                t = body.get("text", "")
                body = {"input": {"transcript": t}, "scenes": [{"text": t}], "look": look_from_spec(asdict(parse(t)))}
                body["look"] = {k: body["look"][k] for k in ("fog", "fps", "grain", "echo", "vignette")}
            run_app_spec(body, os.path.join(out_root, jid), frames=frames, samples=samples)
            JOBS[jid].update(status="done", result=dict(kind="video", src=f"/files/{jid}/clip.mp4",
                                                        poster=f"/files/{jid}/poster.jpg", still=f"/files/{jid}/dream.jpg"))
        except ContentDeclined as e:                     # reported as failed+declined so the current app stops polling and shows the gentle message
            cr = e.rating
            JOBS[jid].update(status="failed", declined=True, error=cr["dreamer_message"], message=cr["dreamer_message"],
                             content_rating={k: v for k, v in cr.items() if k not in ("findings", "abort_reasons")})
        except Exception as e:
            JOBS[jid].update(status="failed", error=str(e), trace=traceback.format_exc()[-2000:])


def make_handler(out_root):
    class H(SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=out_root, **k)

        def end_headers(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "content-type")
            super().end_headers()

        def _json(self, code, obj):
            b = json.dumps(obj).encode()
            self.send_response(code); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

        def do_OPTIONS(self):
            self.send_response(204); self.end_headers()

        def do_POST(self):
            if self.path not in ("/render", "/jobs"):
                return self._json(404, {"error": "not found"})
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            jid = body.get("id") or uuid.uuid4().hex[:10]
            JOBS[jid] = {"status": "queued", "position": Q.qsize()}
            Q.put((jid, body))
            self._json(202, {"job": jid, "job_id": jid, "status_url": f"/jobs/{jid}"})

        def do_GET(self):
            if self.path.startswith("/jobs/"):
                jid = self.path.split("/")[2]
                return self._json(200 if jid in JOBS else 404, JOBS.get(jid, {"error": "unknown job"}))
            if self.path.startswith("/files/"):
                self.path = self.path[len("/files"):]
                return super().do_GET()
            return self._json(200, {"service": "dreamrender", "jobs": len(JOBS)})
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8787); ap.add_argument("--out", default="out/jobs")
    ap.add_argument("--frames", type=int, default=16); ap.add_argument("--samples", type=int, default=32)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    threading.Thread(target=worker, args=(os.path.abspath(a.out), a.frames, a.samples), daemon=True).start()
    print(f"dreamrender listening on :{a.port}")
    ThreadingHTTPServer(("0.0.0.0", a.port), make_handler(os.path.abspath(a.out))).serve_forever()


if __name__ == "__main__":
    main()
