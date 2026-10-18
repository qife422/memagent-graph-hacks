"""
Scripted demo for the hackathon video: two sessions, one graph.

Session 1 — the user teaches the agent.
Session 2 — a FRESH agent process reads the same FalkorDB graph and
            remembers everything. No chat history is passed between
            sessions; the graph is the only memory.
"""

import time

from memory import GraphMemory
from agent import MemAgent


def say(who: str, text: str, delay: float = 0.4):
    print(f"{who}> {text}")
    time.sleep(delay)


def run_demo():
    mem = GraphMemory()
    mem.wipe()
    print("=" * 64)
    print("MemAgent — cross-session memory on FalkorDB (Track 02)")
    print("=" * 64)

    # ---------------- Session 1 ----------------
    print("\n--- SESSION 1 (new user, empty graph) ---\n")
    a1 = MemAgent(user="feifan", session="s1", mem=mem)
    say("you", "I love sci-fi movies")
    for line in a1.learn("I love sci-fi movies"):
        say("memory", line)
    say("you", "I hate horror movies")
    for line in a1.learn("I hate horror movies"):
        say("memory", line)
    say("you", "I'm vegetarian")
    for line in a1.learn("I'm vegetarian"):
        say("memory", line)
    say("agent", a1.answer("what do you remember about me?"))
    print("\n[session 1 ends — process exits, chat history is gone]")

    # ---------------- Session 2 (fresh process, same graph) ----------------
    print("\n--- SESSION 2 (fresh process — only the graph remains) ---\n")
    a2 = MemAgent(user="feifan", session="s2", mem=GraphMemory())
    print(f"[MemAgent] loaded {len(a2.mem.preferences('feifan'))} "
          "facts from FalkorDB, zero chat history\n")
    say("you", "recommend a movie for tonight")
    say("agent", a2.answer("recommend a movie for tonight"))
    say("you", "where should I eat dinner?")
    say("agent", a2.answer("where should I eat dinner?"))
    say("you", "why do you think I like sci-fi?")
    say("agent", a2.answer("why do you think I like sci-fi?"))

    # ---------------- memory update ----------------
    print("\n--- MEMORY UPDATE (preferences change) ---\n")
    say("you", "actually I'm not vegetarian anymore")
    for line in a2.learn("actually I'm not vegetarian anymore"):
        say("memory", line)
    say("you", "where should I eat dinner?")
    say("agent", a2.answer("where should I eat dinner?"))

    print("\n" + "=" * 64)
    print("Done. The graph remembered across sessions; history was superseded,")
    print("never deleted — every belief has a traversable path.")
    print("=" * 64)


if __name__ == "__main__":
    run_demo()
