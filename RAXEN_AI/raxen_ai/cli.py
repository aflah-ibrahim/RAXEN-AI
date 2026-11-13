import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from raxen_ai.agents import build_workflow


def main() -> None:
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY in your environment or .env file first.")

    model = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
    workflow = build_workflow(model)
    state = {"messages": []}

    print("RAXEN AI is ready. Type 'quit' or 'exit' to end the conversation.")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if user_input.lower() in {"quit", "exit"}:
            break
        if not user_input:
            continue

        state = workflow.invoke({**state, "user_input": user_input})
        print(f"RAXEN ({state['agent']}): {state['response']}\n")


if __name__ == "__main__":
    main()