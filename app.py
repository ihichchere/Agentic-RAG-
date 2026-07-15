"""
Point d'entrée principal : interface en ligne de commande pour dialoguer avec
l'agent RAG agentique. La mémoire conversationnelle est activée via un
`thread_id` unique par session (voir commande 'new' pour en changer).

Usage :
    python app.py
"""
import sys
import time
import uuid

from src import config
from src.graph import build_graph

BANNER = """
============================================================
  Agent RAG Agentique — Base documentaire : IA / LLM / RAG
  (LangGraph + Ollama, 100% local, sans clé API)
============================================================
Tapez votre question, ou 'quit' / 'exit' pour quitter.
Tapez 'new' pour démarrer une nouvelle conversation (réinitialise la mémoire).
------------------------------------------------------------
"""


def main() -> None:
    print("Initialisation du graphe agentique (chargement du LLM et de la base vectorielle)...")
    graph = build_graph()
    thread_id = str(uuid.uuid4())
    print(BANNER)

    while True:
        try:
            user_input = input("Vous > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAu revoir !")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            print("Au revoir !")
            break
        if user_input.lower() == "new":
            thread_id = str(uuid.uuid4())
            print("[Nouvelle conversation démarrée — mémoire réinitialisée]\n")
            continue

        run_config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": config.RECURSION_LIMIT,
        }

        t0 = time.time()
        try:
            result = graph.invoke({"messages": [("user", user_input)]}, config=run_config)
        except Exception as exc:
            print(f"\n[Erreur] {exc}\n"
                  "Vérifiez qu'Ollama est bien lancé (`ollama serve`) et que la "
                  "base vectorielle a été construite (`python scripts/build_vectorstore.py`).\n")
            continue
        elapsed = time.time() - t0

        answer = result["messages"][-1].content
        print(f"\nAgent > {answer}")
        print(f"[thread={thread_id[:8]}... | temps de réponse : {elapsed:.2f}s]\n")


if __name__ == "__main__":
    sys.exit(main())
