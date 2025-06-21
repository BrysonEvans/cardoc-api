"""
app.py — CarDoc AI
2025-06-21 patch 10
• FIX: exclude “_no_fault” from DISPLAY lookup (no more KeyError on silent clip)
• All previous features retained
"""

from __future__ import annotations
import importlib, inspect, os, sqlite3, subprocess, sys, tempfile, traceback
from datetime import datetime, timezone
from pathlib import Path

import torch, torchaudio
from openai import OpenAI, OpenAIError
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS

# ────────── ENV / OPENAI ──────────
load_dotenv()
client = OpenAI()
if not os.getenv("OPENAI_API_KEY"):
    sys.exit("❌  OPENAI_API_KEY missing in .env")

# ────────── CONSTANTS ──────────
STAGE1_PTH = Path("models/stage1_engine_detector.pth")
STAGE2_PTH = Path("models/panns_cnn14_checklist_best_aug.pth")
SEARCH_MODULES = ["stage1_model", "pannsupgraded_model", "pannsupgraded"]
CLASS_STAGE1, CLASS_STAGE2 = "Stage1EngineDetector", "PannsChecklist"

DISPLAY = {
    "alternator_whine":  "Alternator Whine",
    "pulley_belt_noise": "Pulley Bearing / Serpentine Belt Noise",
    "rod_knock":         "Rod Knock",
    "timing_chain_rattle": "Timing Chain Rattle",
}
MODEL_LABELS = list(DISPLAY.keys()) + ["engine_idle", "silence"]
IGNORE_LABELS = {"engine_idle", "silence"}

CONF, FLAG = 0.80, 0.60
SILENCE_THRESH = 0.80
TARGET_SR, CLIP_SEC = 32_000, 5

FREE_CHATS_PER_DAY, AD_EVERY = 9, 3
GPT_MODEL, TEMP = "gpt-3.5-turbo", 0.4
MAX_HISTORY, SUMMARY_TRIGGER = 20, 3000
DB_PATH = Path("usage.db")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ────────── FLASK ──────────
app = Flask(__name__)
CORS(app)

# ────────── DB helpers ──────────
def db():
    cn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cn.row_factory = sqlite3.Row
    return cn

with db() as c:
    c.executescript(
        """
        CREATE TABLE IF NOT EXISTS usage(
          device TEXT, date TEXT, count INTEGER,
          subscriber INTEGER DEFAULT 0,
          PRIMARY KEY(device,date)
        );
        CREATE TABLE IF NOT EXISTS chat(
          device TEXT, ts INTEGER, role TEXT, content TEXT
        );
        CREATE TABLE IF NOT EXISTS summary(
          device TEXT PRIMARY KEY, content TEXT
        );
        """
    )

def today(): return datetime.now(timezone.utc).strftime("%Y-%m-%d")

def quota(device:str):
    with db() as c:
        c.execute("INSERT OR IGNORE INTO usage VALUES(?,?,0,0)", (device,today()))
        cnt, sub = c.execute(
            "SELECT count, subscriber FROM usage WHERE device=? AND date=?",
            (device, today())
        ).fetchone()
        if sub: return True, False, cnt          # subscriber = unlimited
        if cnt >= FREE_CHATS_PER_DAY:            # free tier exhausted
            return False, False, cnt
        c.execute("UPDATE usage SET count=count+1 WHERE device=? AND date=?",
                  (device, today()))
        c.commit()
        cnt += 1
        return True, cnt % AD_EVERY == 0, cnt    # ok, maybe show ad

@app.route("/subscribe", methods=["POST"])
def subscribe():
    dev = request.json.get("device_id")
    if not dev:
        return jsonify({"error": "no device_id"}), 400
    with db() as c:
        c.execute("UPDATE usage SET subscriber=1 WHERE device=?", (dev,))
        c.commit()
    return jsonify({"ok": True})

# ────────── AUDIO ──────────
def ffmpeg(src: Path, dst: Path):
    subprocess.check_call(
        ["ffmpeg", "-v", "quiet", "-y", "-i", str(src),
         "-ac", "1", "-ar", str(TARGET_SR), str(dst)]
    )

def wav_tensor(w: Path):
    x, sr = torchaudio.load(str(w))
    if sr != TARGET_SR:
        x = torchaudio.functional.resample(x, sr, TARGET_SR)
    need = CLIP_SEC * TARGET_SR
    if x.size(1) > need:
        x = x[:, :need]
    elif x.size(1) < need:
        x = torch.nn.functional.pad(x, (0, need - x.size(1)))
    return x.to(DEVICE)

# ────────── MODEL loading ──────────
def lazy(cls):
    for m in SEARCH_MODULES:
        try:
            mod = importlib.import_module(m)
            if hasattr(mod, cls):
                return getattr(mod, cls)
        except ModuleNotFoundError:
            pass
    sys.exit(f"{cls} not found")

def load_or_build(path: Path, cls_name: str, labels: int):
    obj = torch.load(path, map_location=DEVICE)
    if hasattr(obj, "eval"):
        return obj.to(DEVICE).eval()
    cls = lazy(cls_name)
    model = cls(**{p: labels for p in inspect.signature(cls).parameters
                   if p in ("num_labels", "num_classes", "classes")}).to(DEVICE)
    model.load_state_dict(obj.get("state_dict", obj), strict=False)
    model.eval()
    return model

stage1 = load_or_build(STAGE1_PTH, CLASS_STAGE1, 2)
stage2 = load_or_build(STAGE2_PTH, CLASS_STAGE2, len(MODEL_LABELS))

@torch.inference_mode()
def stage1_probs(w: Path):
    p = torch.sigmoid(stage1(wav_tensor(w))).squeeze().cpu().numpy()
    if p.ndim == 0:
        return {"engine_idle": float(p), "silence": 1 - float(p)}
    return {"engine_idle": float(p[0]), "silence": float(p[1])}

@torch.inference_mode()
def stage2_probs(w: Path):
    p = torch.sigmoid(stage2(wav_tensor(w))).squeeze().cpu().numpy()
    return {l: round(float(v), 4) for l, v in zip(MODEL_LABELS, p)}

# ────────── /predict ──────────
@app.route("/predict", methods=["POST"])
def predict():
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "no file"}), 400
    with tempfile.TemporaryDirectory() as td:
        raw, wav = Path(td) / "raw", Path(td) / "clip.wav"
        f.save(raw)
        try:
            ffmpeg(raw, wav)
            p1 = stage1_probs(wav)
            if p1["silence"] >= SILENCE_THRESH:
                return jsonify({"silence": 1.0, "_no_fault": True})
            return jsonify(stage2_probs(wav))
        except Exception as e:
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500

# ────────── CHAT memory ──────────
def store(dev, role, text):
    with db() as c:
        c.execute(
            "INSERT INTO chat VALUES(?,?,?,?)",
            (dev, int(datetime.utcnow().timestamp()*1000), role, text)
        )
        c.commit()

def hist(dev):
    with db() as c:
        rows = c.execute(
            "SELECT role, content FROM chat WHERE device=? ORDER BY ts DESC LIMIT ?",
            (dev, MAX_HISTORY)
        ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

def maybe_sum(dev):
    with db() as c:
        chars = c.execute(
            "SELECT SUM(LENGTH(content)) FROM chat WHERE device=?",
            (dev,)
        ).fetchone()[0] or 0
    if chars // 4 < SUMMARY_TRIGGER:
        return
    try:
        summ = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[{"role": "system", "content": "Summarise chat ≤100 words:"}]
                     + hist(dev),
            temperature=0.3,
            max_tokens=120,
        ).choices[0].message.content.strip()
    except OpenAIError:
        return
    with db() as c:
        c.execute("DELETE FROM chat WHERE device=?", (dev,))
        c.execute(
            "INSERT OR REPLACE INTO summary VALUES(?,?)", (dev, summ)
        )
        c.commit()

# ────────── helpers ──────────
def intro(no_fault: bool, faults: str):
    if no_fault:
        return (
            "Hi, I'm CarDoc AI. I didn’t detect a clear engine-sound pattern.\n"
            "Could you share the vehicle’s make & model, describe the noise, "
            "or try recording closer to the sound source?"
        )
    return (
        f"Hi, I'm CarDoc AI. I detected possible {faults}.\n"
        "To narrow this down, could you tell me the vehicle’s make & model "
        "and when/where you hear the noise (cold start, idle, acceleration, etc.)?"
    )

def system_prompt(faults: str):
    return (
        "You are CarDoc AI, an automotive assistant.\n"
        f"Focus on these suspected faults: {faults or 'None'}.\n"
        "Avoid unrelated systems unless the user asks or new evidence appears.\n"
        "Ask short, clear diagnostic questions.\n"
        "Provide only safe inspection steps (vehicle off or safe distance). "
        "Do NOT give hazardous DIY repair instructions. "
        "Do not immediately refer to a professional mechanic unless safety demands it."
    )

# ────────── /gpt-helper ──────────
@app.route("/gpt-helper", methods=["POST"])
def helper():
    d = request.get_json(force=True)
    dev = d.get("device_id") or request.remote_addr or "unknown"
    ok, show_ad, cnt = quota(dev)
    if not ok:
        return jsonify({"error": "quota_exceeded", "limit": FREE_CHATS_PER_DAY}), 402

    scores = d.get("scores", {})
    no_fault = scores.get("_no_fault")
    convo = d.get("conversation") or []
    first_turn = len(convo) == 0
    user_msg = convo[-1].get("text", "") if convo else ""

    faults = [
        DISPLAY[k] for k, v in scores.items()
        if k in DISPLAY and v >= FLAG
    ]
    faults_line = ", ".join(faults)

    # first reply (deterministic, no GPT)
    if first_turn:
        reply = intro(bool(no_fault), faults_line)
        store(dev, "assistant", reply)
        return jsonify({"reply": reply, "show_ad": show_ad, "count": cnt})

    # GPT turn
    ctx = [{"role": "system", "content": system_prompt(faults_line)}]
    with db() as c:
        row = c.execute(
            "SELECT content FROM summary WHERE device=?", (dev,)
        ).fetchone()
    if row:
        ctx.append({"role": "system", "content": "Summary: " + row["content"]})
    ctx += hist(dev)
    ctx.append({"role": "user", "content": user_msg})

    try:
        reply = client.chat.completions.create(
            model=GPT_MODEL,
            messages=ctx,
            temperature=TEMP,
            max_tokens=180,
        ).choices[0].message.content.strip()
    except OpenAIError as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    store(dev, "user", user_msg)
    store(dev, "assistant", reply)
    maybe_sum(dev)
    return jsonify({"reply": reply, "show_ad": show_ad, "count": cnt})

# ────────── run ──────────
if __name__ == "__main__":
    print("🚀  Back-end → http://127.0.0.1:5050")
    app.run(port=5050, debug=False)
