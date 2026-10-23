"""
Graph-backed agent memory powered by FalkorDB.

The agent's memory is a knowledge graph, not a vector blob:
  (User)-[:LIKES {since, session}]->(Item)
  (User)-[:DISLIKES {since, session}]->(Item)
  (User)-[:PREFERS {since, session}]->(Item)

Every memory is a traversable edge with provenance (which session
created it, when). Updates don't delete history — they supersede it,
so the agent can explain *why* it believes something.
"""

import os
from datetime import datetime, timezone

from falkordb import FalkorDB


class GraphMemory:
    def __init__(self, host=None, port=None, graph_name="agent_memory"):
        # Env override so Docker Compose can point at the falkordb service;
        # the defaults keep local and VPS behaviour unchanged.
        host = host or os.environ.get("FALKORDB_HOST", "localhost")
        port = port or int(os.environ.get("FALKORDB_PORT", "6379"))
        self.db = FalkorDB(host=host, port=port)
        self.graph = self.db.select_graph(graph_name)

    # ---------- writes ----------

    def remember(self, user: str, relation: str, item: str,
                 item_type: str = "thing", session: str = "s1") -> None:
        """Store one fact: user -[RELATION]-> item. One edge per triple."""
        now = datetime.now(timezone.utc).isoformat()
        self.graph.query(
            f"""
            MERGE (u:User {{name: $user}})
            MERGE (i:Item {{name: $item}})
            ON CREATE SET i.type = $item_type
            MERGE (u)-[r:{relation}]->(i)
            SET r.since = $now, r.session = $session,
                r.active = true
            """,
            {"user": user, "item": item, "item_type": item_type,
             "now": now, "session": session},
        )

    def forget(self, user: str, relation: str, item: str) -> bool:
        """Supersede a memory: keep history, mark the edge inactive."""
        res = self.graph.query(
            f"""
            MATCH (u:User {{name: $user}})-[r:{relation}]->(i:Item {{name: $item}})
            WHERE r.active IS NULL OR r.active = true
            SET r.active = false, r.superseded_at = $now
            RETURN count(r) AS n
            """,
            {"user": user, "item": item,
             "now": datetime.now(timezone.utc).isoformat()},
        )
        return res.result_set[0][0] > 0

    # ---------- reads ----------

    def preferences(self, user: str) -> list[dict]:
        """All active preferences for a user, with provenance."""
        res = self.graph.query(
            """
            MATCH (u:User {name: $user})-[r:LIKES|DISLIKES|PREFERS]->(i:Item)
            WHERE r.active IS NULL OR r.active = true
            RETURN type(r) AS relation, i.name AS item, i.type AS type,
                   r.since AS since, r.session AS session
            ORDER BY r.since
            """,
            {"user": user},
        )
        return [
            {"relation": row[0], "item": row[1], "type": row[2],
             "since": row[3], "session": row[4]}
            for row in res.result_set
        ]

    def recall(self, user: str, item_type: str | None = None) -> list[dict]:
        """What does the agent know about this user (optionally filtered)?"""
        q = """
            MATCH (u:User {name: $user})-[r]->(i:Item)
            WHERE r.active IS NULL OR r.active = true
        """
        params: dict = {"user": user}
        if item_type:
            q += " AND i.type = $item_type"
            params["item_type"] = item_type
        q += " RETURN type(r) AS relation, i.name AS item ORDER BY r.since"
        res = self.graph.query(q, params)
        return [{"relation": row[0], "item": row[1]} for row in res.result_set]

    def recommend_by_similarity(self, user: str, limit: int = 3) -> list[dict]:
        """Multi-hop: items liked by users who like what you like (Track 01)."""
        res = self.graph.query(
            """
            MATCH (u:User {name: $user})-[:LIKES]->(i:Item)
                  <-[:LIKES]-(other:User)-[:LIKES]->(rec:Item)
            WHERE NOT (u)-[:LIKES]->(rec) AND other.name <> $user
            RETURN rec.name AS item, rec.type AS type, count(*) AS score
            ORDER BY score DESC LIMIT $limit
            """,
            {"user": user, "limit": limit},
        )
        return [
            {"item": row[0], "type": row[1], "score": row[2]}
            for row in res.result_set
        ]

    def explain(self, user: str, item: str) -> list[dict]:
        """Show the graph path behind a belief (explainability)."""
        res = self.graph.query(
            """
            MATCH p = (u:User {name: $user})-[r]->(i:Item {name: $item})
            RETURN type(r) AS relation, r.since AS since,
                   r.session AS session, r.active AS active
            """,
            {"user": user, "item": item},
        )
        return [
            {"relation": row[0], "since": row[1],
             "session": row[2], "active": row[3]}
            for row in res.result_set
        ]

    def wipe(self) -> None:
        """Delete everything (demo reset)."""
        try:
            self.graph.delete()
        except Exception:
            pass  # graph didn't exist yet

    def graph_data(self, user: str = None, at: str = None) -> dict:
        """Dump nodes/edges for visualization.
        user: filter to one user (None = all). at: ISO timestamp — only
        show edges that existed at that time (time travel)."""
        try:
            if at:
                # time travel: edges created at/before `at`, that were still
                # active then (active now, or superseded after `at`)
                res = self.graph.query(
                    "MATCH (u:User)-[r]->(i:Item) "
                    "WHERE r.since <= $at AND "
                    "      (r.active = true OR r.superseded_at > $at OR r.superseded_at IS NULL) "
                    "RETURN u.name, type(r), i.name, i.type, r.active",
                    {"at": at},
                )
            elif user:
                res = self.graph.query(
                    "MATCH (u:User {name: $user})-[r]->(i:Item) "
                    "RETURN u.name, type(r), i.name, i.type, r.active",
                    {"user": user},
                )
            else:
                res = self.graph.query(
                    "MATCH (u:User)-[r]->(i:Item) "
                    "RETURN u.name, type(r), i.name, i.type, r.active"
                )
        except Exception:
            return {"nodes": [], "edges": []}
        nodes, edges, seen = [], [], set()
        for row in res.result_set:
            uname, rel, iname, itype, active = row[0], row[1], row[2], row[3], row[4]
            uid, iid = "u:" + str(uname), "i:" + str(iname)
            if uid not in seen:
                nodes.append({"id": uid, "label": str(uname), "kind": "user"})
                seen.add(uid)
            if iid not in seen:
                nodes.append({"id": iid, "label": str(iname), "kind": "item",
                              "item_type": str(itype or "")})
                seen.add(iid)
            edges.append({"from": uid, "to": iid, "label": str(rel),
                          "active": bool(active) if active is not None else True})
        return {"nodes": nodes, "edges": edges}
