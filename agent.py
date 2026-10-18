"""
MemAgent — a demo agent with cross-session graph memory (Track 02).

The agent itself is deliberately simple: the star of the show is the
FalkorDB memory layer. Every session the agent:
  1. loads the user's active memories from the graph,
  2. answers using them (and shows the Cypher it ran),
  3. stores anything new the user teaches it.

Run:  python agent.py            (interactive)
      python demo.py             (scripted two-session demo for the video)
"""

import re
import sys

from memory import GraphMemory


class MemAgent:
    def __init__(self, user: str = "feifan", session: str = "s1",
                 mem: GraphMemory | None = None):
        self.user = user
        self.session = session
        self.mem = mem or GraphMemory()

    # ---------- tiny NLU: extract "I love X" / "I hate X" / "I prefer X" ----------

    PATTERNS = [
        (re.compile(r"\bi\s+(?:really\s+)?love\s+(.+)", re.I), "LIKES"),
        (re.compile(r"\bi\s+like\s+(.+)", re.I), "LIKES"),
        (re.compile(r"\bi\s+hate\s+(.+)", re.I), "DISLIKES"),
        (re.compile(r"\bi\s+dislike\s+(.+)", re.I), "DISLIKES"),
        (re.compile(r"\bi'?m\s+(?:a\s+)?vegetarian\b", re.I), None),
        (re.compile(r"\bi\s+prefer\s+(.+)", re.I), "PREFERS"),
    ]

    @staticmethod
    def _clean(chunk: str) -> str:
        chunk = chunk.strip().rstrip(".!?").strip()
        return re.sub(r"\s+", " ", chunk)

    def _classify_item(self, item: str) -> str:
        low = item.lower()
        if any(w in low for w in ("movie", "film", "sci-fi", "horror", "comedy",
                                  "drama", "anime")):
            return "genre"
        if any(w in low for w in ("vegetarian", "vegan", "spicy", "sushi",
                                  "pizza", "ramen")):
            return "food"
        if any(w in low for w in ("music", "jazz", "rock", "pop", "classical")):
            return "music"
        return "thing"

    def learn(self, text: str) -> list[str]:
        """Extract memories from user text, store them, return what was learned."""
        learned = []
        low = text.lower()
        if re.search(r"\bi'?m\s+(?:a\s+)?vegetarian\b", low):
            self.mem.remember(self.user, "PREFERS", "vegetarian food",
                              "food", self.session)
            learned.append("PREFERS → vegetarian food")
        if re.search(r"\bi'?m\s+not\s+vegetarian\b", low):
            if self.mem.forget(self.user, "PREFERS", "vegetarian food"):
                learned.append("FORGOT PREFERS → vegetarian food (superseded)")
        for pat, rel in self.PATTERNS:
            m = pat.search(text)
            if m and rel:
                item = self._clean(m.group(1))
                if item and len(item) < 60:
                    self.mem.remember(self.user, rel, item,
                                      self._classify_item(item), self.session)
                    learned.append(f"{rel} → {item}")
        return learned

    # ---------- answering from memory ----------

    def answer(self, text: str) -> str:
        low = text.lower()
        prefs = self.mem.preferences(self.user)

        if any(w in low for w in ("recommend", "suggest")):
            return self._recommend(low, prefs)
        if any(w in low for w in ("movie", "watch", "film", "dinner", "eat",
                                  "food", "lunch", "restaurant")):
            return self._recommend(low, prefs)
        if "what do you remember" in low or "what do you know" in low:
            return self._dump_memory(prefs)
        if "why" in low:
            return self._explain(low, prefs)
        return ("I stored that. Ask me to 'recommend something', or ask "
                "'what do you remember about me?'.")

    def _recommend(self, low: str, prefs: list[dict]) -> str:
        likes = [p for p in prefs if p["relation"] in ("LIKES", "PREFERS")]
        dislikes = [p for p in prefs if p["relation"] == "DISLIKES"]

        if "movie" in low or "watch" in low or "film" in low:
            genres = [p["item"] for p in likes if p["type"] == "genre"]
            bad = [p["item"] for p in dislikes if p["type"] == "genre"]
            sims = self.mem.recommend_by_similarity(self.user)
            extra = f" People with similar taste also like: " + \
                ", ".join(s["item"] for s in sims) if sims else ""
            if genres:
                return (f"Based on your graph memory, you like "
                        f"{', '.join(genres)}"
                        + (f" and dislike {', '.join(bad)}" if bad else "")
                        + f". I'd suggest a {genres[0]} pick tonight.{extra}")
            return "I don't know your movie taste yet — tell me what you love!"
        if any(w in low for w in ("dinner", "eat", "food", "lunch")):
            foods = [p["item"] for p in likes if p["type"] == "food"]
            prefs_food = [p["item"] for p in prefs
                          if p["relation"] == "PREFERS" and p["type"] == "food"]
            bits = list(dict.fromkeys(foods + prefs_food))
            if bits:
                return (f"Your memory says: {', '.join(bits)}. "
                        f"How about a spot that fits?")
            return "I don't know your food preferences yet — tell me!"
        if likes:
            return (f"From memory you like {', '.join(p['item'] for p in likes[:3])}. "
                    f"Want something in that direction?")
        return "I don't remember any preferences yet — teach me what you like!"

    def _dump_memory(self, prefs: list[dict]) -> str:
        if not prefs:
            return "My graph is empty — I remember nothing about you yet."
        lines = ["Here's what my FalkorDB memory holds about you:"]
        for p in prefs:
            lines.append(f"  • {p['relation']}: {p['item']} "
                         f"(since {p['since'][:10]}, session {p['session']})")
        return "\n".join(lines)

    def _explain(self, low: str, prefs: list[dict]) -> str:
        if not prefs:
            return "I have no memories to explain yet."
        target = None
        for p in prefs:
            words = [w.strip(".,!?\"'") for w in p["item"].lower().split()]
            words = [w for w in words if len(w) > 3]
            if any(w in low for w in words):
                target = p
                break
        p = target or prefs[-1]
        paths = self.mem.explain(self.user, p["item"])
        info = paths[0] if paths else {}
        return (f"I believe '{p['item']}' because of this graph path:\n"
                f"  (User:{self.user})-[{p['relation']}]->"
                f"(Item:{p['item']})\n"
                f"  learned in session {info.get('session')}, "
                f"at {str(info.get('since'))[:19]}")


def interactive():
    mem = GraphMemory()
    session = input("Session id (e.g. s1): ").strip() or "s1"
    agent = MemAgent(session=session, mem=mem)
    print(f"\n[MemAgent] session={session} — "
          f"memory loaded: {len(agent.mem.preferences(agent.user))} facts.")
    print("Teach me ('I love sci-fi'), ask ('recommend a movie'), "
          "or 'quit'.\n")
    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in ("quit", "exit"):
            break
        learned = agent.learn(text)
        for line in learned:
            print(f"  [memory] {line}")
        if not learned or not text.lower().startswith("i "):
            print("agent>", agent.answer(text))
    print("\n[MemAgent] session saved to FalkorDB. See you next session.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        from demo import run_demo
        run_demo()
    else:
        interactive()
