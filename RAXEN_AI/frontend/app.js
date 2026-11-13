const API_URL =
  window.location.port === "5500"
    ? "http://127.0.0.1:8000/api/chat/stream"
    : "/api/chat/stream";

const form = document.querySelector("#chat-form");
const input = document.querySelector("#message-input");
const sendButton = document.querySelector("#send-button");
const conversation = document.querySelector("#conversation");
const welcome = document.querySelector("#welcome");
const status = document.querySelector("#status");
const messages = [];
let pending = false;
let handoffTimer;
let handoffNotice;

function isNearConversationBottom() {
  const page = document.documentElement;
  return page.scrollHeight - window.innerHeight - window.scrollY < 120;
}

function scrollToConversationBottom() {
  window.scrollTo(0, document.documentElement.scrollHeight);
}

function addMessage(role, content, agent = "") {
  const article = document.createElement("article");
  article.className = `message ${role}`;

  let label;
  if (role === "assistant") {
    label = document.createElement("p");
    label.className = "message-label";
    label.textContent = `RAXEN · ${agent || "Assistant"}`;
    article.append(label);
  }

  const body = document.createElement("div");
  body.className = "message-content";
  body.textContent = content;
  article.append(body);
  conversation.append(article);
  article.scrollIntoView({ behavior: "smooth", block: "nearest" });
  return { article, body, label };
}

function showHandoffNotice(agentName) {
  clearTimeout(handoffTimer);
  handoffNotice?.remove();

  const agent = agentName[0].toUpperCase() + agentName.slice(1);
  handoffNotice = document.createElement("p");
  handoffNotice.className = "handoff-notice";
  handoffNotice.setAttribute("role", "status");
  handoffNotice.textContent = `Handing off to ${agent}…`;
  conversation.append(handoffNotice);
  handoffNotice.scrollIntoView({ behavior: "smooth", block: "nearest" });
  handoffTimer = setTimeout(() => {
    handoffNotice?.remove();
    handoffNotice = undefined;
  }, 2000);
}

function setPending(value) {
  pending = value;
  sendButton.disabled = value;
  input.disabled = value;
  sendButton.setAttribute("aria-busy", String(value));
}

async function sendMessage(value) {
  const prompt = value.trim();
  if (!prompt || pending) return;

  welcome.hidden = true;
  status.textContent = "";
  status.classList.remove("error", "handoff");
  clearTimeout(handoffTimer);
  handoffNotice?.remove();
  handoffNotice = undefined;
  const previousMessages = messages.slice();
  const userMessage = addMessage("user", prompt);
  const assistantMessage = addMessage("assistant", "", "Typing…");
  messages.push({ role: "user", content: prompt });
  setPending(true);
  status.textContent = "Finding the right RAXEN specialist…";

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: previousMessages, prompt }),
    });

    if (!response.ok) {
      const result = await response.json();
      throw new Error(result.detail || `Request failed (${response.status}).`);
    }
    if (!response.body) throw new Error("Streaming is unavailable in this browser.");

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let complete = false;

    const handleEvent = (rawEvent) => {
      const data = rawEvent
        .split("\n")
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trimStart())
        .join("\n");
      if (!data) return;

      const event = JSON.parse(data);
      if (event.type === "delta") {
        const shouldFollow = isNearConversationBottom();
        assistantMessage.body.textContent += event.content;
        if (shouldFollow) scrollToConversationBottom();
      } else if (event.type === "handoff") {
        showHandoffNotice(event.agent);
      } else if (event.type === "done") {
        assistantMessage.label.textContent = `RAXEN · ${event.agent}`;
        messages.push({
          role: "assistant",
          content: assistantMessage.body.textContent,
        });
        complete = true;
      } else if (event.type === "error") {
        throw new Error(event.message);
      }
    };

    while (true) {
      const { value: chunk, done } = await reader.read();
      buffer += decoder.decode(chunk, { stream: !done }).replace(/\r\n/g, "\n");
      const events = buffer.split("\n\n");
      buffer = events.pop() || "";
      for (const event of events) handleEvent(event);
      if (done) break;
    }
    if (buffer.trim()) handleEvent(buffer);
    if (!complete) throw new Error("The response stream ended unexpectedly.");
    status.textContent = "";
  } catch (error) {
    userMessage.article.remove();
    assistantMessage.article.remove();
    messages.splice(0, messages.length, ...previousMessages);
    welcome.hidden = previousMessages.length > 0;
    input.value = prompt;
    input.style.height = "auto";
    input.style.height = `${Math.min(input.scrollHeight, 150)}px`;
    status.textContent = `${error.message} Your message is back in the composer so you can retry it.`;
    status.classList.add("error");
  } finally {
    setPending(false);
    input.focus();
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  const prompt = input.value;
  input.value = "";
  input.style.height = "auto";
  sendMessage(prompt);
});

input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 150)}px`;
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

document.querySelectorAll("[data-prompt]").forEach((button) => {
  button.addEventListener("click", () => sendMessage(button.dataset.prompt));
});

document.querySelector("#new-chat").addEventListener("click", () => {
  if (pending) return;
  clearTimeout(handoffTimer);
  handoffNotice?.remove();
  handoffNotice = undefined;
  messages.length = 0;
  conversation.replaceChildren();
  status.textContent = "";
  status.classList.remove("error", "handoff");
  welcome.hidden = false;
  input.value = "";
  input.focus();
});
