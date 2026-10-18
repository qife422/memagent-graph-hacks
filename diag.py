"""Diagnostic: why does recall return [] in session 2?"""
from memory import GraphMemory
from agents import Team
from llm import LLM

print("=== Direct read via new connection ===")
m = GraphMemory()
prefs = m.preferences("feifan")
print(f"direct preferences(): {len(prefs)} facts")
for p in prefs:
    print(" ", p["relation"], "->", p["item"])

print("\n=== Via Team wiring ===")
team = Team(GraphMemory(), user="feifan", session="sX", llm=LLM())
print("team.mem is GraphMemory:", type(team.mem).__name__)
print("advisor_runner.user:", repr(team.advisor_runner.user))
print("advisor_runner.mem is team.mem:", team.advisor_runner.mem is team.mem)
print("advisor recall via runner:",
      team.advisor_runner.run("recall", {}))

print("\n=== Graph object identity ===")
m1 = GraphMemory()
m2 = GraphMemory()
print("m1.graph is m2.graph:", m1.graph is m2.graph)
print("m1.db is m2.db:", m1.db is m2.db)
