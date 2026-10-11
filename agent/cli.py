from __future__ import annotations

from dotenv import load_dotenv

from agent.runtime import build_graph, build_prompt, load_reinforcement_files, make_chat_model, run_turn
from clinic.fixtures import make_session


def main() -> None:
    load_dotenv()
    patches = load_reinforcement_files()
    prompt = build_prompt(patches)
    session = make_session()
    model = make_chat_model()
    graph = build_graph(model)
    print("Northwind Clinic scheduler. Type 'quit' to exit.")
    if patches:
        print("Active reinforcements:", ", ".join(patch["id"] for patch in patches))
    else:
        print("No reinforcements loaded. This is the base policy.")
    while True:
        try:
            user_text = input("You: ").strip()
        except EOFError:
            print()
            break
        if not user_text:
            continue
        if user_text.lower() in {"quit", "exit"}:
            break
        reply = run_turn(session, user_text, prompt, graph)
        print(f"Agent: {reply}")
        if session.terminal:
            print(f"(phase: {session.phase})")


if __name__ == "__main__":
    main()
