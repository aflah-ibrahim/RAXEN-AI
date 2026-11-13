import base64
import binascii
import hmac
import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from raxen_ai.agents import build_workflow


logger = logging.getLogger(__name__)
load_dotenv()
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(title="RAXEN AI API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5500", "http://127.0.0.1:5500"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.middleware("http")
async def require_public_access(request, call_next):
    password = os.getenv("PUBLIC_ACCESS_PASSWORD")
    if password:
        scheme, _, encoded_credentials = request.headers.get(
            "authorization", ""
        ).partition(" ")
        try:
            credentials = base64.b64decode(
                encoded_credentials, validate=True
            ).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError):
            credentials = ""

        username, separator, supplied_password = credentials.partition(":")
        if (
            scheme.lower() != "basic"
            or not separator
            or username != "raxen"
            or not hmac.compare_digest(supplied_password, password)
        ):
            return JSONResponse(
                status_code=401,
                content={"detail": "Authentication required."},
                headers={"WWW-Authenticate": 'Basic realm="RAXEN AI", charset="UTF-8"'},
            )

    return await call_next(request)


class ConversationMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[ConversationMessage]
    prompt: str


class ChatResponse(BaseModel):
    agent: str
    response: str


@lru_cache(maxsize=1)
def _get_workflow():
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="The server is missing OPENAI_API_KEY. Configure it in .env.",
        )
    model = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
    )
    return build_workflow(model)


def _to_langchain_messages(messages: list[ConversationMessage]) -> list[BaseMessage]:
    return [
        HumanMessage(content=message.content)
        if message.role == "user"
        else AIMessage(content=message.content)
        for message in messages
    ]


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    prompt = request.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=422, detail="Message cannot be empty.")

    try:
        result = _get_workflow().invoke(
            {
                "messages": _to_langchain_messages(request.messages),
                "user_input": prompt,
            }
        )
    except HTTPException:
        raise
    except Exception as error:
        logger.exception("Chat workflow failed")
        raise HTTPException(
            status_code=502,
            detail="The assistant could not respond. Please try again.",
        ) from error

    return ChatResponse(agent=result["agent"], response=result["response"])


@app.post("/api/chat/stream")
def stream_chat(request: ChatRequest) -> StreamingResponse:
    prompt = request.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=422, detail="Message cannot be empty.")
    workflow = _get_workflow()
    workflow_input = {
        "messages": _to_langchain_messages(request.messages),
        "user_input": prompt,
    }

    def events():
        try:
            for event in workflow.stream(workflow_input, stream_mode="custom"):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception:
            logger.exception("Streaming chat workflow failed")
            error = {"type": "error", "message": "The assistant could not respond. Please try again."}
            yield f"data: {json.dumps(error)}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


if __name__ == "__main__":
    uvicorn.run("raxen_ai.api:app", host="127.0.0.1", port=8000)
