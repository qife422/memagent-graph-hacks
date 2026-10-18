"""
Agent tools — the function surface the LLM can call.
Every tool reads or writes the FalkorDB graph. This is the "coordination"
layer: Scout and Advisor share state exclusively through these tools.
"""

from memory import GraphMemory


def tool_definitions(exclude=None):
    exclude = set(exclude or [])
    all_tools = [
        {
            "type": "function",
            "function": {
                "name": "remember",
                "description": ("Store a fact about the user in graph memory. "
                                "relation is LIKES, DISLIKES or PREFERS."),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "relation": {"type": "string",
                                     "enum": ["LIKES", "DISLIKES", "PREFERS"]},
                        "item": {"type": "string"},
                        "item_type": {"type": "string",
                                      "description": "genre, food, music..."},
                    },
                    "required": ["relation", "item"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "recall",
                "description": ("Read the user's active memories from the graph. "
                                "Call this FIRST before answering anything. "
                                "item_type is an optional hint, e.g. 'movie'."),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "item_type": {"type": "string"},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "forget",
                "description": "Mark a memory as superseded (history is kept).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "relation": {"type": "string"},
                        "item": {"type": "string"},
                    },
                    "required": ["relation", "item"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "recommend_similar",
                "description": ("Multi-hop graph query: items liked by users "
                                "who like what this user likes."),
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "explain",
                "description": "Show the graph path behind a belief about an item.",
                "parameters": {
                    "type": "object",
                    "properties": {"item": {"type": "string"}},
                    "required": ["item"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "ask_scout",
                "description": ("Ask the Scout agent to learn something new from "
                                "the user and store it. Use ONLY when recall "
                                "shows the memory is truly missing."),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                    },
                    "required": ["question"],
                },
            },
        },
    ]
    return [t for t in all_tools if t["function"]["name"] not in exclude]


class ToolRunner:
    def __init__(self, mem, user, session, scout_callback=None):
        self.mem = mem
        self.user = user
        self.session = session
        self.scout_callback = scout_callback
        self.log = []

    def run(self, name, args):
        if name == "remember":
            self.mem.remember(self.user, args["relation"], args["item"],
                              args.get("item_type", "thing"), self.session)
            out = "remembered: %s -> %s" % (args["relation"], args["item"])
        elif name == "recall":
            prefs = self.mem.preferences(self.user)
            if args.get("item_type"):
                want = args["item_type"].lower()
                prefs = [p for p in prefs
                         if want in p["type"].lower()
                         or p["type"].lower() in want]
            shown = ["%s:%s" % (p["relation"], p["item"]) for p in prefs]
            out = str(shown) if shown else "no memories"
        elif name == "forget":
            ok = self.mem.forget(self.user, args["relation"], args["item"])
            out = ("superseded " + args["item"]) if ok else "not found"
        elif name == "recommend_similar":
            recs = self.mem.recommend_by_similarity(self.user)
            out = str(recs) if recs else "no similar-taste data yet"
        elif name == "explain":
            paths = self.mem.explain(self.user, args["item"])
            out = str(paths) if paths else "no path found"
        elif name == "ask_scout":
            if self.scout_callback:
                out = self.scout_callback(args["question"])
            else:
                out = "scout unavailable"
        else:
            out = "unknown tool: " + name
        self.log.append("  [tool:%s] %s" % (name, out[:120]))
        return out
