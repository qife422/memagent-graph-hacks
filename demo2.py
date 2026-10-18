"""
MemAgent v2 — deep demo: LLM brains + two coordinating agents, one graph.

  SCOUT    learns preferences through natural conversation (LLM + tools)
  ADVISOR  answers from graph memory (LLM + tools)
  GRAPH    FalkorDB — the only shared state. Agents never see each
           other's chat history; coordination happens via ask_scout
           handoffs and the graph itself.

Run:
  export OPENAI_API_KEY='sk-...' && python demo2.py
"""

import sys
import time

from memory import GraphMemory
from llm import LLM
from agents import Team


def say(who: str, text: str, delay: float = 0.5):
    print(f"\n{who}> {text}")
    time.sleep(delay)


def trace(team: Team):
    for line in team.tool_trace():
        print(line)


def main():
    llm = LLM()
    if not llm.available:
        print("Set OPENAI_API_KEY first: export OPENAI_API_KEY='sk-...'")
        sys.exit(1)

    mem = GraphMemory()
    mem.wipe()

    print("=" * 70)
    print("MemAgent v2 — two LLM agents, one FalkorDB graph (Track 02)")
    print("=" * 70)

    # ---------- Session 1: Scout learns via natural conversation ----------
    print("\n--- SESSION 1: Scout learns ---\n")
    team1 = Team(mem, user="feifan", session="s1", llm=llm)

    say("you", "Hey, I'm really into sci-fi movies lately, especially space operas")
    say("scout", team1.scout.chat(
        "Hey, I'm really into sci-fi movies lately, especially space operas"))
    trace(team1)

    say("you", "I can't stand horror though. And I'm vegetarian.")
    say("scout", team1.scout.chat(
        "I can't stand horror though. And I'm vegetarian."))
    trace(team1)

    print("\n[session 1 ends — both agents' chat histories are wiped]")

    # ---------- Session 2: Advisor answers from graph alone ----------
    print("\n--- SESSION 2: Advisor answers (fresh, graph only) ---\n")
    team2 = Team(GraphMemory(), user="feifan", session="s2", llm=llm)
    n = len(team2.mem.preferences("feifan"))
    print(f"[both agents boot with {n} graph facts, zero chat history]\n")

    say("you", "Recommend a movie for tonight")
    say("advisor", team2.advisor.chat("Recommend a movie for tonight"))
    trace(team2)

    say("you", "Why do you think I'd like that?")
    say("advisor", team2.advisor.chat("Why do you think I'd like that?"))
    trace(team2)

    # ---------- Coordination: Advisor -> Scout -> graph -> Advisor ----------
    print("\n--- COORDINATION: missing memory triggers a Scout handoff ---\n")
    say("you", "What music should I put on?")

    # Scripted stand-in for the human: when Scout asks, the "user" replies.
    # In live use a real human answers here.
    def scripted_scout(question: str) -> str:
        print(f"\nscout> {question} (asking the user…)")
        time.sleep(0.5)
        user_says = "I love jazz, especially Miles Davis"
        say("you", user_says)
        reply = team2.scout.chat(
            f"Advisor asked: {question}\nUser answered: {user_says}\n"
            f"Store what you learned with the remember tool, then confirm.")
        trace(team2)
        return f"Scout stored it: {reply}"

    team2.advisor_runner.scout_callback = scripted_scout
    say("advisor", team2.advisor.chat("What music should I put on?"))
    trace(team2)

    say("you", "OK now — what music should I put on?")
    say("advisor", team2.advisor.chat("What music should I put on?"))
    trace(team2)

    print("\n" + "=" * 70)
    print("Done. Two agents, zero shared chat history — the graph was the")
    print("only memory, and every belief is a traversable, explainable edge.")
    print("=" * 70)


if __name__ == "__main__":
    main()
