"""CLI entry point for the multi-agent enterprise documentation assistant."""
from agents.manager import ManagerAgent


def format_response(response) -> str:
    lines = [f"\n[{response.query_type.value.upper()}]", response.answer]
    citations = response.details.get("citations")
    if citations:
        lines.append("\nSources:")
        for c in citations:
            lines.append(f"  - {c.source} (chunk {c.chunk_index}, similarity {c.similarity:.2f})")
    sql = response.details.get("sql")
    if sql:
        lines.append(f"\nSQL: {sql}")
        lines.append(f"Rows returned: {len(response.details.get('rows') or [])}")
    return "\n".join(lines)


def main():
    print("Enterprise Documentation Assistant — type 'exit' to quit.\n")
    manager = ManagerAgent()
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        try:
            response = manager.handle(question)
        except Exception as e:
            print(f"\n[ERROR] Couldn't answer that: {e}\n")
            continue

        print(format_response(response))

if __name__ == "__main__":
    main()