"""HyDE RAG Chatbot — CLI entry point.

Run:
    cd hyde_rag
    python main.py

Commands:
    /quit   — exit the chatbot
    /clear  — reset conversation history
    /sources — show sources from the last retrieval
"""

from __future__ import annotations

import uuid

from dotenv import load_dotenv

load_dotenv()  # Load .env before anything reads settings

from graph.graph_builder import compile_graph  # noqa: E402


def main():
    print("=" * 60)
    print("  HyDE RAG Chatbot  (type /quit to exit)")
    print("=" * 60)
    print()

    app = compile_graph()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    last_sources: list[dict] = []

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue

        # ── Slash commands ──────────────────────────────────────
        if query.lower() == "/quit":
            print("Goodbye!")
            break

        if query.lower() == "/clear":
            thread_id = str(uuid.uuid4())
            config = {"configurable": {"thread_id": thread_id}}
            print("[System] Conversation history cleared.\n")
            continue

        if query.lower() == "/sources":
            if not last_sources:
                print("[System] No sources from a previous query.\n")
            else:
                print("[Sources]")
                for s in last_sources:
                    print(f"  - {s['source']} (score: {s['score']:.3f})")
                print()
            continue

        # ── Run the HyDE graph ─────────────────────────────────
        result = app.invoke({"query": query}, config=config)

        # Check for errors
        if result.get("error"):
            print(f"\n[Error] {result['error']}\n")
            continue

        # Show the hypothetical doc (for transparency / debugging)
        hyp = result.get("hypothetical_doc", "")
        if hyp:
            # Truncate display to first 200 chars
            display = hyp[:200] + ("..." if len(hyp) > 200 else "")
            print(f"\n[HyDE Doc]: {display}")

        # Show retrieved sources
        docs = result.get("retrieved_docs", [])
        last_sources = []
        if docs:
            print(f"\n[Retrieved {len(docs)} source(s)]")
            for doc in docs:
                src = doc.metadata.get("source", "unknown")
                score = doc.metadata.get("score", 0)
                last_sources.append({"source": src, "score": score})
                print(f"  - {src} (score: {score:.3f})")

        # Final answer
        answer = result.get("answer", "No answer generated.")
        print(f"\nAssistant: {answer}\n")


if __name__ == "__main__":
    main()
