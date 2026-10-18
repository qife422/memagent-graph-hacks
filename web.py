"""
MemAgent web UI — visual demo for Graph Hacks (Track 02).

  python web.py  →  http://localhost:5000

Needs: FalkorDB on localhost:6379, OPENAI_API_KEY env var.
"""

import os
import time
import uuid
from flask import Flask, jsonify, render_template, request

from memory import GraphMemory
from llm import LLM
from agents import Team
from proactive import ProactiveEngine

app = Flask(__name__)
llm = LLM()

# Global team; "new session" swaps it for a fresh one on the same graph.
# _state["user"] is the currently viewed user (switchable via /api/user).
_state = {"team": None, "session": None, "chat": [], "user": "feifan"}


def _get_team_for_proactive():
    return get_team()


proactive_engine = ProactiveEngine(_get_team_for_proactive, llm,
                                   enabled=True, poll_seconds=60)
proactive_engine.start()


def get_team():
    if _state["team"] is None or _state["team"].user != _state["user"]:
        _state["session"] = "web-" + uuid.uuid4().hex[:6]
        _state["team"] = Team(GraphMemory(), user=_state["user"],
                               session=_state["session"], llm=llm)
    return _state["team"]


@app.route("/")
def index():
    return render_template("index.html",
                           llm_ok=llm.available,
                           session=_state["session"] or "-",
                           user=_state["user"])


@app.route("/api/chat", methods=["POST"])
def chat():
    if not llm.available:
        return jsonify({"error": "Set OPENAI_API_KEY"}), 400
    data = request.get_json(force=True)
    agent_name = data.get("agent", "advisor")
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"error": "empty message"}), 400
    team = get_team()
    agent = team.scout if agent_name == "scout" else team.advisor
    t0 = time.time()
    reply = agent.chat(message)
    trace = team.tool_trace()
    team.scout_runner.log.clear()
    team.advisor_runner.log.clear()
    entry = {"agent": agent_name, "user": message, "reply": reply,
             "trace": trace, "ms": int((time.time() - t0) * 1000)}
    _state["chat"].append(entry)
    entry["graph"] = team.mem.graph_data(user=_state["user"])
    entry["session"] = _state["session"]
    return jsonify(entry)


@app.route("/api/graph")
def graph():
    user = request.args.get("user") or _state["user"]
    at = request.args.get("at")  # ISO timestamp for time travel
    return jsonify(get_team().mem.graph_data(user=user, at=at))


@app.route("/api/users")
def users():
    """List all users that have memories in the graph."""
    mem = get_team().mem
    try:
        res = mem.graph.query("MATCH (u:User) RETURN DISTINCT u.name ORDER BY u.name")
        names = [row[0] for row in res.result_set]
    except Exception:
        names = []
    if "feifan" not in names:
        names = ["feifan"] + names
    return jsonify({"users": names, "current": _state["user"]})


@app.route("/api/user", methods=["POST"])
def switch_user():
    data = request.get_json(force=True)
    name = (data.get("name") or "feifan").strip() or "feifan"
    _state["user"] = name
    _state["team"] = None  # force rebuild with new user
    _state["chat"] = []
    team = get_team()
    return jsonify({"user": name, "session": _state["session"],
                    "graph": team.mem.graph_data(user=name)})


@app.route("/api/session/new", methods=["POST"])
def new_session():
    # Fresh agents, same graph: proves memory lives in FalkorDB, not chat.
    _state["session"] = "web-" + uuid.uuid4().hex[:6]
    _state["team"] = Team(GraphMemory(), user=_state["user"],
                            session=_state["session"], llm=llm)
    _state["chat"] = []
    n = len(_state["team"].mem.preferences(_state["user"]))
    return jsonify({"session": _state["session"],
                    "facts_in_graph": n,
                    "graph": _state["team"].mem.graph_data(user=_state["user"])})


@app.route("/api/graph/reset", methods=["POST"])
def reset_graph():
    get_team().mem.wipe()
    return jsonify({"graph": get_team().mem.graph_data()})


@app.route("/api/explain")
def explain():
    item = request.args.get("item", "")
    user = request.args.get("user") or _state["user"]
    paths = get_team().mem.explain(user, item)
    return jsonify({"item": item, "paths": paths})


@app.route("/api/insight", methods=["POST"])
def insight():
    """Proactive insight: Advisor analyzes the user's taste profile
    and volunteers an observation + recommendation, unprompted."""
    if not llm.available:
        return jsonify({"error": "Set OPENAI_API_KEY"}), 400
    team = get_team()
    prefs = team.mem.preferences(_state["user"])
    if not prefs:
        return jsonify({"insight": "I don't know your tastes yet — switch to Scout and teach me a few things first!"})
    # Build a taste summary for the LLM
    likes = [p["item"] for p in prefs if p["relation"] == "LIKES"]
    dislikes = [p["item"] for p in prefs if p["relation"] == "DISLIKES"]
    prompt = (
        f"You are Advisor, a proactive AI with graph memory. "
        f"User {_state['user']} likes: {', '.join(likes) or 'nothing yet'}. "
        f"Dislikes: {', '.join(dislikes) or 'nothing yet'}. "
        f"Volunteer ONE sharp observation about their taste pattern "
        f"(1-2 sentences), then ONE specific recommendation they'd love "
        f"that they haven't mentioned. Be concrete, no hedging. "
        f"Format: 👀 Observation: ...\n💡 Try: ..."
    )
    try:
        msg = llm.chat([{"role": "user", "content": prompt}])
        text = (msg.get("content") or "").strip() or "Hmm, I drew a blank — try again."
    except Exception as e:
        text = f"Insight failed: {e}"
    return jsonify({"insight": text})


@app.route("/api/seed", methods=["POST"])
def seed():
    """Load rich demo data for alice and bob (multi-user showcase)."""
    mem = get_team().mem
    demo = {
        "alice": [
            ("LIKES", "sci-fi movies"), ("LIKES", "jazz"), ("LIKES", "sushi"),
            ("DISLIKES", "horror"), ("PREFERS", "vegetarian food"),
            ("LIKES", "Murakami"),
        ],
        "bob": [
            ("LIKES", "horror movies"), ("LIKES", "metal"), ("LIKES", "steak"),
            ("DISLIKES", "jazz"), ("PREFERS", "action movies"),
            ("LIKES", "Tarantino"),
        ],
    }
    counts = {}
    for user, prefs in demo.items():
        n = 0
        for rel, item in prefs:
            try:
                mem.remember(user, rel, item, session="seed",
                             item_type="demo")
                n += 1
            except Exception:
                pass
        counts[user] = n
    return jsonify({"alice": counts.get("alice", 0),
                    "bob": counts.get("bob", 0)})


@app.route("/api/proactive/pending")
def proactive_pending():
    """Frontend polls this; returns queued proactive recommendations."""
    items = proactive_engine.pending()
    # attach current graph so the UI can refresh
    if items:
        try:
            graph = get_team().mem.graph_data(user=_state["user"])
        except Exception:
            graph = None
        return jsonify({"items": items, "graph": graph})
    return jsonify({"items": []})


@app.route("/api/proactive/trigger", methods=["POST"])
def proactive_trigger():
    """Force one proactive judgment cycle (demo/testing)."""
    if not llm.available:
        return jsonify({"error": "Set OPENAI_API_KEY"}), 400
    items = proactive_engine.trigger_now()
    graph = None
    if items:
        try:
            graph = get_team().mem.graph_data(user=_state["user"])
        except Exception:
            pass
    return jsonify({"items": items, "graph": graph})


@app.route("/api/proactive/toggle", methods=["POST"])
def proactive_toggle():
    data = request.get_json(force=True, silent=True) or {}
    proactive_engine.enabled = bool(data.get("enabled", True))
    return jsonify({"enabled": proactive_engine.enabled})


@app.route("/api/timeline")
def timeline():
    """All memories (active + superseded) ordered by time."""
    user = request.args.get("user") or _state["user"]
    mem = get_team().mem
    try:
        res = mem.graph.query(
            "MATCH (u:User {name: $user})-[r]->(i:Item) "
            "RETURN type(r) AS rel, i.name AS item, r.since AS since, "
            "r.session AS session, r.active AS active "
            "ORDER BY r.since",
            {"user": user},
        )
        items = [
            {"relation": row[0], "item": row[1], "since": str(row[2] or ""),
             "session": row[3],
             "active": bool(row[4]) if row[4] is not None else True}
            for row in res.result_set
        ]
    except Exception:
        items = []
    return jsonify({"items": items})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("MemAgent web UI on http://localhost:%d (LLM: %s)"
          % (port, "ok" if llm.available else "MISSING KEY"))
    app.run(host="0.0.0.0", port=port, threaded=True)
