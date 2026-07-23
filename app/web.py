

import os
import sys
import json
import signal
import threading
from pathlib import Path

import flask
from flask import Flask, request, jsonify, render_template

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import main
from connect_to_llm import list_sessions, delete_session


from config import WEB_HOST, WEB_PORT

app = Flask(__name__)

_runtime_started = False
_startup_lock = threading.Lock()

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/query", methods=["POST"])
def api_query():
    data = request.get_json(force=True)
    query = data.get("query", "").strip()
    session_id = data.get("session_id", "default")
    top_k = int(data.get("top_k", 100))

    if not query:
        return jsonify({"error": "Missing 'query' field"}), 400

    try:
        answer = main.get_answer(query, top_k=top_k, session_id=session_id,
                                 agentic=data.get("agentic", "full"))
        chunks = main.get_chunks(query, top_k=top_k,
                                 agentic=data.get("agentic", "full"))
        safe_chunks = []
        for c in chunks:
            safe_chunks.append({
                k: v for k, v in c.items()
                if isinstance(v, (str, int, float, bool, type(None)))
            })
        return jsonify({"answer": answer, "chunks": safe_chunks})
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {e}"}), 500


@app.route("/api/sessions", methods=["GET"])
def api_list_sessions():
    return jsonify({"sessions": list_sessions()})


@app.route("/api/sessions", methods=["POST"])
def api_create_session():
    data = request.get_json(force=True) or {}
    session_id = data.get("session_id", "").strip()
    if not session_id:
        # Generate a short unique name
        import uuid
        session_id = f"session-{uuid.uuid4().hex[:8]}"
    # Touch it so it exists in the store
    from connect_to_llm import get_session_history
    get_session_history(session_id)
    return jsonify({"session_id": session_id}), 201


@app.route("/api/sessions/<session_id>", methods=["DELETE"])
def api_delete_session(session_id):
    delete_session(session_id)
    return jsonify({"status": "deleted"})


def start_runtime():
    global _runtime_started
    with _startup_lock:
        if _runtime_started:
            return
        print("Starting RAG runtime (Weaviate + reranker)...")
        main.run_app(rebuild_index=False, kill_existing_reranker=True)
        _runtime_started = True
        print("RAG runtime ready.")


def shutdown(signum, frame):
    print("\nShutting down...")
    main.close_app()
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Start runtime before Flask (blocking startup)
    start_runtime()

    print(f"Web UI → http://{WEB_HOST}:{WEB_PORT}")
    app.run(host=WEB_HOST, port=WEB_PORT, threaded=True, debug=False)
