# MemAgent — Dual Agents Sharing One Graph Memory

> **Graph Hacks (WeMakeDevs × FalkorDB) · Track 02: Agent Memory & Coordination**
>
> 🎥 Demo video: https://youtu.be/5b57-C2ra_M
> 🚀 Live demo: http://129.146.75.2:8000 (runs on Oracle Cloud Always Free ARM)
> 📝 Blog post: [TODO — optional, for the Best Blog prize]

Two AI agents. One shared brain. **MemAgent** is a pair of cooperating agents — **Scout** and **Advisor** — that read from and write to the same [FalkorDB](https://github.com/FalkorDB/FalkorDB) knowledge graph as their long-term memory. Scout learns what you like from conversation. Advisor remembers it next week, explains *why* it remembers, and — when it's genuinely worth your time — speaks up on its own.

## The idea in 30 seconds

Chatbots forget. Every new session starts from zero, and "personalization" usually means a hidden vector store nobody can inspect. MemAgent takes a different bet:

- **Memory you can see.** Every preference Scout learns becomes a typed edge in a graph (`LIKES`, `DISLIKES`) — rendered live, traversable, and explainable. When Advisor recommends something, it can show you the exact path through the graph that led there.
- **Memory that updates like a human's.** Preferences aren't deleted; they're *superseded*. Change your mind about horror movies and the old edge is marked inactive, not erased — so you get a full time-travel history of how your taste evolved.
- **Memory that works for you, not just when asked.** A background scheduler periodically asks a simple question — *"given the time, the weather, and everything in this graph, is there something worth saying right now?"* — and stays silent unless the answer is yes. No notification spam, no fake engagement.

## Live demo walkthrough (the 6-step one-click Demo)

Press **▶ Demo** and watch the whole story in ~2 minutes:

| Step | What happens | What it proves |
|------|--------------|----------------|
| 1–3 | Chat with **Scout**; it picks up preferences (movies, food, music…) | Learning → graph writes, watch nodes appear live |
| 4 | Start a **new session** — chat history is wiped | The graph remembers what the chat forgot |
| 5 | Hit **💡 Insights** — **Advisor** analyzes your taste pattern | Cross-session recall + proactive analysis |
| 6 | **✨ Proactive nudge** — MemAgent speaks up unprompted (toast + chat message) | Memory working *for* you, not just *when asked* |

No API key? Steps 1–4 and the graph visualization work without one; LLM features (Scout/Advisor chat, Insights, nudges) need `OPENAI_API_KEY`.

## How it works

```
  ┌─────────┐   learns from chat    ┌──────────────────┐
  │  Scout  │ ───────────────────▶  │                  │
  └─────────┘                       │  FalkorDB graph  │
                                    │  (typed edges:   │
  ┌─────────┐   recalls + explains  │   LIKES/DISLIKES)│
  │ Advisor │ ◀───────────────────  │                  │
  └─────────┘                       └──────────────────┘
        │                                    ▲
        │  proactive scheduler                 │
        │  (11:30 / 15:00 / 20:00 local)       │
        ▼                                    │
  ┌─────────────┐    LLM gate: "worth saying?"  │
  │ toast + chat│ ── yes ──▶ speaks up          │
  │ (or silence)│ ── no ───▶ stays quiet        │
  └─────────────┘
```

**The proactive loop** (the part we're proudest of):
1. **Gather context** — local time, weather (Open-Meteo, free, no key needed), and a summary of the memory graph.
2. **Ask the gate** — an LLM judge decides whether anything is *worth* saying. The prompt is deliberately conservative: *"don't invent shop names, don't pester."*
3. **Speak or stay silent** — if yes, a toast slides in and an Advisor message lands in the chat, citing the memory behind it ("*With the overcast weather, a warm bowl of vegetarian ramen could hit the spot…*"). If no, nothing happens. Most checks end in silence — that's the point.

**Memory model** (honest version): preferences are stored as typed, timestamped edges between a user node and item nodes, with `active` flags for supersede-not-delete updates. Explanations are produced by traversing the edges that fired — no black box. Multi-user separation in the demo is via an explicit user switcher (each user's subgraph is filtered); the production path is one graph per user (`graph_name=f"memory_{user_id}"`) plus a minimal login — see Roadmap.

## Graph data model

Two node labels, one relationship pattern:

- **`(u:User {name})`** — one node per user; this is what isolates memory between users (the multi-user switcher just switches the `User` node).
- **`(i:Item {name, type})`** — anything the agent learns about: a preference, a food, a meeting slot…
- **`[r:LIKES|DISLIKES|PREFERS|…]`** — typed relationship `User → Item`, carrying the memory itself:
  - `r.since` — when the fact was learned
  - `r.session` — which session learned it
  - `r.active` — `true`/`NULL` = current; `false` = superseded (never deleted)
  - `r.superseded_at` — when it was superseded

Updates are **supersede-not-delete**: a changed preference deactivates the old edge
instead of removing it, so the full history stays traversable (the time-travel slider
in the UI replays it via `r.since` / `r.superseded_at`).

## Key Cypher queries

**Learn a fact** — merge the user, the item, and the typed edge:
```cypher
MERGE (u:User {name: $user})
MERGE (i:Item {name: $item})
ON CREATE SET i.type = $item_type
MERGE (u)-[r:LIKES]->(i)
SET r.since = $now, r.session = $session,
    r.active = true
```

**Supersede (not delete) an outdated fact** — the old edge stays in the graph:
```cypher
MATCH (u:User {name: $user})-[r:LIKES]->(i:Item {name: $item})
WHERE r.active IS NULL OR r.active = true
SET r.active = false, r.superseded_at = $now
RETURN count(r) AS n
```

**Recall active preferences** — what the Advisor agent reasons over:
```cypher
MATCH (u:User {name: $user})-[r:LIKES|DISLIKES|PREFERS]->(i:Item)
WHERE r.active IS NULL OR r.active = true
RETURN type(r) AS relation, i.name AS item, i.type AS type,
       r.since AS since, r.session AS session
ORDER BY r.since
```

**Graph-based recommendation** — collaborative filtering over the graph
(2-hop: people who like what you like → what else they like):
```cypher
MATCH (u:User {name: $user})-[:LIKES]->(i:Item)<-[:LIKES]-(other:User)
      -[:LIKES]->(rec:Item)
WHERE NOT (u)-[:LIKES]->(rec) AND other.name <> $user
RETURN rec.name AS item, rec.type AS type, count(*) AS score
ORDER BY score DESC
```

**Explain a memory** — traversable provenance for any single fact:
```cypher
MATCH p = (u:User {name: $user})-[r]->(i:Item {name: $item})
RETURN type(r) AS relation, r.since AS since,
       r.session AS session, r.active AS active
```

**Time-travel** — replay the graph as it looked at time `$at`:
```cypher
MATCH (u:User)-[r]->(i:Item)
WHERE r.since <= $at AND
      (r.superseded_at IS NULL OR r.superseded_at > $at)
RETURN u.name, type(r), i.name, i.type, r.active
```

All queries run against FalkorDB (Redis-protocol graph DB) via `redis-py`.

## Why Track 02

Agent memory is usually the boring part of the stack — a vector DB you never look at. This entry argues it should be the *interesting* part: a coordination substrate two agents genuinely share. Scout writes, Advisor reads, and the handoff between them is visible in the UI (watch the handoff visualization during the demo). The graph isn't storage; it's the medium the agents think *through* — and because it's a graph, every recommendation carries its own receipt.

## Tech stack

- **Graph:** FalkorDB (Docker) — the memory substrate
- **Backend:** Python (`web.py`) — chat API, Insights, proactive scheduler
- **LLM:** OpenAI `gpt-4o-mini` — cheap enough to run the proactive gate 3–4×/day for pennies
- **Frontend:** vanilla JS + canvas graph rendering — dark/light themes, fullscreen graph, physics & edge-label toggles
- **Hosting:** Oracle Cloud Always Free (ARM, 2 OCPU / 12 GB)

## Run it yourself

```bash
# 1. FalkorDB
docker run -p 6379:6379 -p 3000:3000 falkordb/falkordb:latest

# 2. App (needs Python 3.10+)
git clone https://github.com/qife422/memagent-graph-hacks.git
cd memagent-graph-hacks
pip install -r requirements.txt

# 3. Your own key — whoever runs it, brings their own
export OPENAI_API_KEY="sk-..."
PORT=8000 python3 web.py
# → http://localhost:8000
```

## Roadmap (post-hackathon)

1. **Real multi-tenancy** — one graph per user + minimal username/password login (the demo switcher becomes real isolation).
2. **Richer proactive triggers** — calendar awareness, location-optional nudges.
3. **Memory types** — explicit episodic / semantic / procedural layers on top of the edge model.
4. **Friends trial** — deploy with a shared key behind a simple allowlist before any public launch.

What we're *not* doing: pretending the demo switcher is production auth, or bolting on paid APIs (e.g. real restaurant names need a Places API — the demo honestly stays at cuisine/dish level).

## Built for

**Graph Hacks** — WeMakeDevs × FalkorDB, October 15–18, 2026 · Track 02: Agent Memory & Coordination.
Solo build. Every line of the MemAgent app was written during the event window; FalkorDB does what it does best underneath.
