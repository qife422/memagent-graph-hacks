"""
Two agents, one graph.

SCOUT — the learner. Talks to the user, extracts preferences with the LLM,
writes them to FalkorDB via tools.

ADVISOR — the reasoner. Answers questions by reading graph memory via tools.
When memory is missing, it hands off to Scout (ask_scout), Scout learns,
writes to the graph, and Advisor re-reads. Coordination happens entirely
through the shared graph — the agents never pass chat history to each other.
"""

import json
from llm import LLM
from tools import tool_definitions, ToolRunner


SCOUT_SYSTEM = """You are Scout, a preference-learning agent. Your job: talk to the user,
figure out what they like/dislike/prefer, and store it with the remember tool.
Be conversational and brief. When you've stored what you learned, summarize.
Only use the remember tool — you don't answer questions, you learn."""

ADVISOR_SYSTEM = """You are Advisor, a recommendation agent with graph memory.
RULE 1: ALWAYS call recall FIRST, before any other tool, for every user question.
RULE 2: Only if recall returns 'no memories' for what they ask, use ask_scout.
RULE 3: Ground every answer in what the graph says. Cite the memory you used.
You may use recommend_similar for multi-hop taste data and explain to show
the graph path behind a belief. Be brief."""


class Agent:
    def __init__(self, name, system, llm, runner, session, exclude_tools=None):
        self.name = name
        self.system = system
        self.llm = llm
        self.runner = runner
        self.session = session
        self.exclude_tools = exclude_tools or []
        self.history = [{"role": "system", "content": system}]

    def _tool_loop(self, user_text, max_steps=6):
        self.history.append({"role": "user", "content": user_text})
        tools = tool_definitions(exclude=self.exclude_tools)
        for _ in range(max_steps):
            msg = self.llm.chat(self.history, tools=tools)
            self.history.append(msg)
            calls = msg.get("tool_calls") or []
            if not calls:
                return msg.get("content", "")
            for call in calls:
                fn = call["function"]
                args = json.loads(fn["arguments"] or "{}")
                result = self.runner.run(fn["name"], args)
                self.history.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": result,
                })
        return "(max tool steps reached)"

    def chat(self, text):
        if not self.llm.available:
            return "(LLM unavailable — set OPENAI_API_KEY)"
        return self._tool_loop(text)


class Team:
    """Scout + Advisor sharing one FalkorDB graph."""

    def __init__(self, mem, user="feifan", session="s1", llm=None):
        self.mem = mem
        self.user = user
        self.session = session
        self.llm = llm or LLM()

        self.scout_runner = ToolRunner(mem, user, session)
        self.scout = Agent("Scout", SCOUT_SYSTEM, self.llm,
                           self.scout_runner, session,
                           exclude_tools=["ask_scout"])

        def scout_handoff(question):
            reply = self.scout.chat(
                "The Advisor needs this answered and stored: " + question)
            return "Scout: " + reply

        self.advisor_runner = ToolRunner(mem, user, session,
                                         scout_callback=scout_handoff)
        self.advisor = Agent("Advisor", ADVISOR_SYSTEM, self.llm,
                             self.advisor_runner, session)

    def tool_trace(self):
        return self.scout_runner.log + self.advisor_runner.log
