"""
Proactive recommendations — MemAgent volunteers suggestions unprompted,
like a good assistant would.

How it works (mirrors how Muse does proactivity):
  1. Gather context: time of day, day of week, weather (Open-Meteo, free),
     and the user's taste profile from the FalkorDB memory graph.
  2. Judgment gate: ask the LLM "is there something worth proactively
     recommending right now? Be conservative — silence is fine."
  3. Deliver: if yes, queue the recommendation; the frontend polls and
     shows a toast + inserts an Advisor message.

Trigger moments (not too frequent — 3x/day):
  11:30 → lunch idea     15:00 → afternoon pick-me-up     20:00 → evening wind-down

Weather: Open-Meteo (https://open-meteo.com), free, no API key.
Location defaults to Corvallis, OR (user's college town); override with
PROACTIVE_LAT / PROACTIVE_LON env vars.
"""

import json
import os
import threading
import time
import urllib.request
from datetime import datetime
from zoneinfo import ZoneInfo

# All candidate moments are in the user's timezone (America/Los_Angeles),
# NOT server local time — the VPS runs on UTC.
LA = ZoneInfo("America/Los_Angeles")


def _now_la():
    return datetime.now(LA)

# (moment_name, hour, minute) — local time
CANDIDATE_MOMENTS = [
    ("lunch", 11, 30),
    ("afternoon", 15, 0),
    ("evening", 20, 0),
]

# Corvallis, OR — override via env
LAT = float(os.environ.get("PROACTIVE_LAT", "44.5646"))
LON = float(os.environ.get("PROACTIVE_LON", "-123.2620"))

WEATHER_CODE_MEANING = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "foggy", 48: "icy fog", 51: "light drizzle", 53: "drizzle",
    55: "heavy drizzle", 61: "light rain", 63: "rain", 65: "heavy rain",
    71: "light snow", 73: "snow", 75: "heavy snow",
    80: "light showers", 81: "showers", 82: "violent showers",
    95: "thunderstorm", 96: "thunderstorm with hail", 99: "thunderstorm with hail",
}


def get_weather():
    """Current weather via Open-Meteo (free, no key). Returns a short string."""
    try:
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={LAT}&longitude={LON}"
            "&current=temperature_2m,weathercode&timezone=auto"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "MemAgent/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        cur = data.get("current", {})
        temp = cur.get("temperature_2m")
        code = cur.get("weathercode")
        desc = WEATHER_CODE_MEANING.get(code, "unknown")
        if temp is None:
            return desc
        return f"{desc}, {round(temp)}°C"
    except Exception:
        return "unknown"


def build_context(user, mem):
    """Gather everything the judgment gate needs."""
    now = _now_la()
    prefs = []
    try:
        prefs = mem.preferences(user)
    except Exception:
        pass
    likes = [p["item"] for p in prefs if p["relation"] == "LIKES"]
    dislikes = [p["item"] for p in prefs if p["relation"] == "DISLIKES"]
    prefers = [p["item"] for p in prefs if p["relation"] == "PREFERS"]
    return {
        "moment": now.strftime("%A %H:%M"),
        "weekday": now.strftime("%A"),
        "is_weekend": now.weekday() >= 5,
        "weather": get_weather(),
        "likes": likes,
        "dislikes": dislikes,
        "prefers": prefers,
        "has_taste": bool(prefs),
    }


JUDGE_PROMPT = """You are Advisor, a proactive AI assistant with graph memory of the user.
Current context:
- Time: {moment} ({weekday}{weekend_note})
- Weather: {weather}
- User likes: {likes}
- User dislikes: {dislikes}
- User prefers: {prefers}

Decide: is there something genuinely worth volunteering RIGHT NOW — unprompted?
Good triggers: weekend lunch idea matching their taste, a song for a nice afternoon,
an evening wind-down pick, something tied to weather + their preferences.
Be conservative. If nothing feels natural, say NOTHING.

Rules:
- Recommend a TYPE of thing (cuisine, dish, genre, artist, activity) — never invent
  specific restaurant names, addresses, or prices you can't verify.
- One short message, warm tone, under 60 words. No hedging.
- If not worth interrupting, reply with exactly: SKIP

Your reply (either SKIP or the recommendation message):"""


def judge_and_generate(user, mem, llm):
    """Returns a recommendation string, or None if the gate says stay silent."""
    if not llm.available:
        return None
    ctx = build_context(user, mem)
    if not ctx["has_taste"]:
        return None  # nothing to personalize with yet
    prompt = JUDGE_PROMPT.format(
        moment=ctx["moment"],
        weekday=ctx["weekday"],
        weekend_note=", weekend" if ctx["is_weekend"] else "",
        weather=ctx["weather"],
        likes=", ".join(ctx["likes"]) or "nothing yet",
        dislikes=", ".join(ctx["dislikes"]) or "nothing yet",
        prefers=", ".join(ctx["prefers"]) or "nothing yet",
    )
    try:
        msg = llm.chat([{"role": "user", "content": prompt}])
        text = (msg.get("content") or "").strip()
    except Exception:
        return None
    if not text or text.strip().upper().startswith("SKIP"):
        return None
    return text


class ProactiveEngine:
    """Background scheduler. Frontend polls pending() for new recommendations."""

    def __init__(self, get_team_fn, llm, enabled=True, poll_seconds=60):
        self._get_team = get_team_fn
        self._llm = llm
        self.enabled = enabled
        self.poll_seconds = poll_seconds
        self._pending = []          # queued recommendations not yet delivered
        self._fired = set()         # (date_str, moment_name) already triggered
        self._lock = threading.Lock()
        self._thread = None

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def _loop(self):
        while True:
            try:
                self._tick()
            except Exception:
                pass  # never kill the scheduler on an error
            time.sleep(self.poll_seconds)

    def _tick(self):
        if not self.enabled:
            return
        now = _now_la()
        today = now.strftime("%Y-%m-%d")
        for name, hour, minute in CANDIDATE_MOMENTS:
            key = (today, name)
            # fire once, within 10 minutes after the moment
            after = (now.hour * 60 + now.minute) - (hour * 60 + minute)
            if 0 <= after < 10 and key not in self._fired:
                with self._lock:
                    if key in self._fired:
                        continue
                    self._fired.add(key)
                self._fire(name)

    def _fire(self, moment_name):
        try:
            team = self._get_team()
            user = team.user
            text = judge_and_generate(user, team.mem, self._llm)
        except Exception as e:
            print(f"[proactive] _fire failed for {moment_name}: {e}")
            return
        if text:
            with self._lock:
                self._pending.append({
                    "moment": moment_name,
                    "text": text,
                    "at": _now_la().isoformat(),
                })
        else:
            print(f"[proactive] gate said SKIP for {moment_name} (user={user})")

    def pending(self):
        """Pop all undelivered recommendations."""
        with self._lock:
            items = self._pending[:]
            self._pending.clear()
            return items

    def trigger_now(self, moment_name="manual"):
        """Force one judgment cycle (for demo / testing).
        Queues the recommendation WITHOUT popping, so the frontend
        polling picks it up and shows the toast — full end-to-end flow."""
        self._fire(moment_name)
        # peek without popping: frontend will fetch via /api/proactive/pending
        with self._lock:
            return list(self._pending)
