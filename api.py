import os
import traceback

import httpx
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request
from fastapi.responses import PlainTextResponse
from groq import RateLimitError
from pydantic import BaseModel

from sugar2 import graph

app = FastAPI()
API_SECRET = os.environ.get("BOT_SECRET", "")
IG_VERIFY_TOKEN = os.environ.get("IG_VERIFY_TOKEN", "")
IG_ACCESS_TOKEN = os.environ.get("IG_ACCESS_TOKEN", "")


class Incoming(BaseModel):
    user_id: str
    message: str


def get_reply(user_id: str, message: str) -> str:
    config = {"configurable": {"thread_id": user_id}}
    try:
        result = graph.invoke(
            {"messages": [("user", message)], "stage": "menu"},
            config,
        )
        reply = result["messages"][-1].content
        return reply or "Sorry, could you please say that again?"

    except RateLimitError:
        return "We're getting a lot of messages right now. Our team will reply to you shortly 🧁"

    except Exception:
        traceback.print_exc()
        return "Sorry, something went wrong on our side. Our team will help you shortly 🧁"


@app.post("/chat")
def chat(body: Incoming, x_api_key: str = Header(default="")):
    if API_SECRET and x_api_key != API_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return {"reply": get_reply(body.user_id, body.message)}


# ---------- Instagram webhook ----------

@app.get("/webhook")
def verify_webhook(request: Request):
    p = request.query_params
    if p.get("hub.mode") == "subscribe" and p.get("hub.verify_token") == IG_VERIFY_TOKEN:
        return PlainTextResponse(p.get("hub.challenge", ""))
    raise HTTPException(status_code=403, detail="Verification failed")


def handle_instagram_message(sender_id: str, text: str):
    reply = get_reply(f"ig_{sender_id}", text)
    try:
        r = httpx.post(
            "https://graph.instagram.com/v21.0/me/messages",
            headers={"Authorization": f"Bearer {IG_ACCESS_TOKEN}"},
            json={"recipient": {"id": sender_id}, "message": {"text": reply[:1000]}},
            timeout=15,
        )
        print("IG send:", r.status_code, r.text)
    except Exception:
        traceback.print_exc()


@app.post("/webhook")
async def receive_webhook(request: Request, background_tasks: BackgroundTasks):
    data = await request.json()
    print("IG webhook:", data)
    for entry in data.get("entry", []):
        for event in entry.get("messaging", []):
            msg = event.get("message", {})
            text = msg.get("text")
            if not text or msg.get("is_echo"):
                continue
            background_tasks.add_task(handle_instagram_message, event["sender"]["id"], text)
    return {"status": "ok"}