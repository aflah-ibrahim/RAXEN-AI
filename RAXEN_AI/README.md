# RAXEN AI

RAXEN AI is a shopping-assistant chat app with a plain HTML, CSS, and JavaScript
frontend and a separate FastAPI backend. The backend uses a LangGraph workflow
to route conversations between three specialists:

- **Adrian Marcus** — gaming PCs, workstations, and custom configurations.
- **Product Concierge** — accessories and other IT products.
- **Ethan Cole** — purchase preparation and checkout guidance.

The router selects a specialist using the conversation context. Any specialist
can hand off to another when that better fits the request. Handoffs are limited
to three per message to prevent routing loops.

Adrian has three in-memory example workstation configurations with illustrative
USD price estimates. They are not verified RAXEN listings, current prices, or
inventory. The app does not connect to live product catalogs, inventory,
databases, carts, or checkout. The frontend keeps conversation history in
browser memory; refreshing the page or starting a new conversation clears it.

## Requirements

- Python 3.11 or newer (Python 3.11 is the tested version)
- An OpenAI API key

The frontend uses browser-native HTML, CSS, and JavaScript; it does not require
Node.js or a frontend build step. Newer Python versions may need additional
build tools if a dependency does not yet provide a compatible wheel.

## Install

From the project root, create and activate a virtual environment, install the
project, and create a local environment file:

```bash
python3.11 -m venv .venv3.11
source .venv3.11/bin/activate
python -m pip install -e '.[dev]'
cp .env.example .env
```

If `.venv3.11` already exists, do not recreate it; activate it and continue with
the install command. If `.env` already exists, keep it and skip the copy command
so you do not overwrite your local settings. Set `OPENAI_API_KEY` in `.env` or in
your shell. `OPENAI_MODEL` is optional and defaults to `gpt-4o-mini`.

Keep `.env` private; it is excluded from version control.

## Run the chat app

Start each server in a separate terminal, with the project root as the working
directory.

**Terminal 1 — API backend:**

```bash
source .venv3.11/bin/activate
python -m raxen_ai.api
```

The API listens on `http://127.0.0.1:8000`. Check that it is running at
<http://127.0.0.1:8000/api/health>.

**Terminal 2 — static frontend:**

```bash
python3 -m http.server 5500 --directory frontend
```

Open <http://localhost:5500> in your browser. New user messages appear
immediately, and assistant text streams in as it is generated. The frontend sends
chat requests to the local API; keep both servers running while using the app.
Stop either server with `Ctrl+C`.

The API also serves the frontend from `http://127.0.0.1:8000/`, which is useful
for same-origin deployments and tunnels. Set `PUBLIC_ACCESS_PASSWORD` to enable
HTTP Basic authentication for the frontend and every API endpoint; the username
is `raxen`. Leave it unset for local development only.

### Temporary public sharing

For a temporary public link, run the API with a strong access password, then
point Cloudflare Tunnel at it:

```bash
export PUBLIC_ACCESS_PASSWORD='use-a-long-random-password'
.venv3.11/bin/python -m uvicorn raxen_ai.api:app --host 127.0.0.1 --port 8001
cloudflared tunnel --url http://127.0.0.1:8001
```

Share the generated `https://*.trycloudflare.com` URL and the password with the
intended visitor using a separate channel. The link is temporary and only works
while both processes and this computer remain online. Chat traffic passes
through Cloudflare; requests to the AI provider use the server's private API key.

## Run the command-line chat

With the virtual environment activated and `.env` configured:

```bash
python -m raxen_ai.cli
```

Enter `quit` or `exit` to end the conversation.

## API

- `GET /api/health` returns `{"status":"ok"}`.
- `POST /api/chat/stream` accepts the conversation so far and latest prompt,
  then streams server-sent events (`delta`, followed by `done`):

  ```json
  {
    "messages": [
      {"role": "user", "content": "I need a PC."},
      {"role": "assistant", "content": "What will you use it for?"}
    ],
    "prompt": "A custom build for gaming."
  }
  ```

  Events use the `text/event-stream` content type. For example:

  ```text
  data: {"type":"delta","content":"Let's "}

  data: {"type":"delta","content":"plan your build."}

  data: {"type":"done","agent":"adrian"}
  ```

- `POST /api/chat` accepts the same request and returns the final specialist and
  response as JSON for non-streaming clients.

The API does not persist conversations between requests. The client supplies
conversation history with each chat request.

## Tests

Run the test suite from the project root with the virtual environment activated:

```bash
pytest
```

Tests use mock models and do not make OpenAI API calls.
