# 🧁 Sugar Bubbles Bot

An AI-powered ordering assistant for **Sugar Bubbles**, a home-based bakery selling cookies, brownies, brownie tubs, and cakes. Customers chat with the bot to browse the menu, ask questions, and place orders.

Built with **LangGraph**, **Groq (LLM)**, and **FastAPI**.

## Features

- Answers menu, flavor, and pricing questions from a single source-of-truth menu
- Understands assorted boxes (mix any flavors in one box)
- Remembers each customer's conversation (per-user memory)
- Tool calling through LangGraph's `ToolNode`
- REST API with an optional API-key check
- Graceful handling of rate limits and errors, so customers never see a crash

## Tech Stack

| Part | Tool |
|------|------|
| Agent framework | LangGraph |
| LLM | Groq via `langchain-groq` |
| API | FastAPI + Uvicorn |
| Memory | LangGraph `MemorySaver` (per-user thread) |
| Config | python-dotenv |

## Project Structure

~~~
api.py            # FastAPI server (/chat endpoint)
sugar2.py         # LangGraph agent: menu data, tools, graph
sugar2.ipynb      # Notebook used to build and test the agent
requirements.txt  # Python dependencies
~~~

## Getting Started

**1. Clone and install**

~~~bash
git clone https://github.com/mohdmustafa084/sugar-bubbles-bot.git
cd sugar-bubbles-bot
pip install -r requirements.txt
~~~

**2. Add your keys.** Create a `.env` file in the project folder:

~~~
GROQ_API_KEY=your_groq_key_here
BOT_SECRET=optional_api_key_for_the_chat_endpoint
~~~

**3. Run the server**

~~~bash
uvicorn api:app --port 8000
~~~

**4. Try it.** Open http://127.0.0.1:8000/docs, expand `POST /chat`, and send:

~~~json
{ "user_id": "customer1", "message": "What cookie flavors do you have?" }
~~~

If you set `BOT_SECRET`, send it in the `x-api-key` header.

## API

`POST /chat`

| Field | Type | Description |
|-------|------|-------------|
| `user_id` | string | Unique customer ID, used for conversation memory |
| `message` | string | The customer's message |

Response: `{ "reply": "..." }`

## Roadmap

- [ ] Instagram DM integration (webhook + Meta Graph API)
- [ ] Save confirmed orders to a sheet or database
- [ ] Persistent memory (replace in-memory storage)
- [ ] Deploy to a public URL

## Author

**Mohd Mustafa** · [GitHub](https://github.com/mohdmustafa084)