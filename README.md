# Agent RAG Agentique — Base documentaire IA / LLM (LangGraph + Ollama)

Projet réalisé dans le cadre de l'évaluation de fin de module **Agentic RAG**
(Master IIBDCC, Prof. RETAL Sara). Système de question-réponse agentique
construit avec **LangGraph** (graphe `StateGraph` écrit à la main, sans
`create_agent`/`create_react_agent`) sur un corpus de 10 articles
scientifiques de référence en IA / Machine Learning, avec un LLM et des
embeddings servis **100% localement par Ollama** (aucune clé API requise).

## 1. Pourquoi ce corpus ?

Les 10 articles couvrent les fondations des LLM et du RAG agentique
lui-même (Transformer, BERT, GPT-3, InstructGPT, RAG, Chain-of-Thought,
LoRA, ReAct, Self-RAG, Llama 2) — voir `scripts/download_papers.py` pour la
liste complète avec les identifiants arXiv. Ce choix rend les questions
d'évaluation naturelles à vérifier et donne un fil rouge cohérent pour la
vidéo de démonstration.

## 2. Architecture du système

```
        START ──▶ agent ──[tool_calls ?]──▶ tools ──[outil = retrieval ?]──▶ grade_documents
                    ▲ non                                │ non                    │
                    │                                     ▼                        │
                   END                                  agent            ┌─────────┴─────────┐
                                                                          │ pertinent          │ non pertinent
                                                                          ▼                    │ (retry < max)
                                                                      generate ◀── retry épuisé ┤
                                                                          │                     ▼
                                                                          ▼               rewrite_query
                                                                         END                     │
                                                                          ▲                       │
                                                                          └────── agent ◀─────────┘
```

- **`agent`** : le LLM (avec les outils liés) décide s'il répond directement
  ou s'il doit appeler un outil — c'est la décision agentique centrale.
- **`tools`** : exécute l'outil choisi (`retrieve_documents`,
  `list_available_papers` ou `get_paper_summary`).
- **`grade_documents`** : un appel LLM dédié juge si les documents
  récupérés sont réellement pertinents (logique inspirée de *Corrective
  RAG* / *Self-RAG*) — le système n'accepte jamais un retrieval à l'aveugle.
- **`rewrite_query`** : si les documents ne sont pas pertinents, reformule
  la requête et relance une recherche (boucle bornée par
  `MAX_QUERY_REWRITES` pour éviter toute boucle infinie).
- **`generate`** : génère la réponse finale, ancrée sur les documents
  validés, avec citation des sources.

La **mémoire conversationnelle** (multi-tours) est gérée par un
`MemorySaver` (checkpointer LangGraph) indexé par `thread_id` — voir
`src/graph.py` et `app.py`. Le détail de chaque nœud est commenté dans
`src/graph.py`.

Pour régénérer le diagramme exact de votre graphe compilé (recommandé pour
le rapport et la vidéo) : `python scripts/visualize_graph.py` (voir
§5). Un aperçu est aussi fourni dans `outputs/graph.mmd`.

## 3. Structure du projet

```
agentic-rag-cs-papers/
├── app.py                       # CLI interactif (point d'entrée principal)
├── requirements.txt
├── .env.example                 # config par défaut (à copier en .env)
├── data/pdfs/                   # les 10 articles arXiv (téléchargés par le script)
├── src/
│   ├── config.py                # configuration centrale
│   ├── llm.py                   # wrappers ChatOllama / OllamaEmbeddings
│   ├── ingestion.py              # chargement PDF -> chunks -> ChromaDB
│   ├── tools.py                  # les 3 outils exposés à l'agent
│   ├── state.py                  # schéma du state LangGraph
│   └── graph.py                  # le graphe agentique (cœur du projet)
├── scripts/
│   ├── download_papers.py       # télécharge les 10 PDFs depuis arXiv
│   ├── build_vectorstore.py     # prétraitement + vectorisation (ChromaDB)
│   ├── visualize_graph.py       # génère le diagramme du graphe
│   └── run_evaluation.py        # évaluation sur 20 questions (10 + 10)
├── evaluation/
│   ├── questions.json           # 10 questions simples + 10 complexes
│   └── results/                 # généré par run_evaluation.py
└── outputs/                     # graphe visualisé, résultats
```

## 4. Installation

**Prérequis : Python 3.10+ et [Ollama](https://ollama.com) installé.**

```bash
# 1. Cloner le dépôt et se placer à la racine
cd agentic-rag-cs-papers

# 2. Créer un environnement virtuel
python3 -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Copier la configuration par défaut
cp .env.example .env

# 5. Démarrer Ollama (dans un terminal séparé, ou en service selon l'OS)
ollama serve

# 6. Télécharger les modèles locaux utilisés par le projet
ollama pull llama3.1
ollama pull nomic-embed-text
```

> Machine modeste (pas de GPU) ? Remplacez `llama3.1` par un modèle plus
> léger dans `.env` (`CHAT_MODEL=llama3.2` ou `mistral`), puis
> `ollama pull llama3.2`.

## 5. Utilisation — à exécuter dans cet ordre

Toutes les commandes ci-dessous s'exécutent **depuis la racine du projet**.

```bash
# Étape 1 — Construction de la base documentaire (télécharge les 10 PDFs)
python scripts/download_papers.py

# Étape 2 — Prétraitement et vectorisation (chunking + embeddings + ChromaDB)
python scripts/build_vectorstore.py

# Étape 3 — Dialoguer avec l'agent
python app.py

# Étape 4 (optionnel) — Visualiser le graphe (image + code Mermaid)
python scripts/visualize_graph.py

# Étape 5 — Évaluation complète (10 questions simples + 10 complexes)
python scripts/run_evaluation.py
```

`run_evaluation.py` écrit deux fichiers dans `evaluation/results/` :
- `eval_details.json` (réponse complète, sources, temps, score par question)
- `eval_results.csv` (tableau récapitulatif, facile à importer dans le rapport)

**Ces fichiers contiennent les vrais résultats de VOTRE exécution locale —
ce sont les chiffres à reporter dans la section "Résultats" du rapport.**

## 6. Où voir chaque critère de la grille d'évaluation

| Critère de la grille              | Où le voir dans ce projet |
|-----------------------------------|----------------------------|
| Prétraitement et vectorisation    | `src/ingestion.py`, `scripts/build_vectorstore.py` |
| Développement des outils          | `src/tools.py` (3 outils, docstrings = description function-calling) |
| Qualité du graphe LangGraph       | `src/graph.py` + `outputs/graph.mmd` |
| Respect de l'approche Agentic RAG | boucle `grade_documents` → `rewrite_query` dans `src/graph.py` (décision, auto-correction, tool use) |
| Qualité du code                   | structure `src/`/`scripts/`, docstrings, séparation des responsabilités |
| Expérimentation et Simulation     | `scripts/run_evaluation.py`, `evaluation/questions.json` |
| Rapport                           | voir le fichier `.docx` fourni séparément |

## 7. Dépannage rapide

- **`FileNotFoundError: Aucun PDF trouvé`** → lancez
  `python scripts/download_papers.py` avant `build_vectorstore.py`.
- **Erreur de connexion Ollama** → vérifiez qu'`ollama serve` tourne bien
  (`curl http://localhost:11434/api/tags` doit répondre).
- **Modèle introuvable** → vérifiez `ollama list` et que le nom dans `.env`
  correspond exactement (ex: `llama3.1`, pas `llama3.1:latest` sauf si
  c'est le tag réellement présent).
- **Réponses lentes** → normal sur CPU pour un modèle 7-8B ; réduisez
  `TOP_K` dans `.env`, ou utilisez un modèle plus petit (`llama3.2:3b`).
- **PNG du graphe non généré** → nécessite un accès internet à l'API
  mermaid.ink ; le fichier `outputs/graph.mmd` (texte) fonctionne toujours
  et peut être collé sur https://mermaid.live pour export manuel.

## 8. Limites connues et pistes d'amélioration

Voir la section dédiée du rapport (`.docx`). En bref : mémoire uniquement
"court terme" (pas de mémoire long-terme cross-session), LLM local plus
limité en raisonnement qu'un modèle propriétaire de pointe, absence
d'outil de recherche web pour sortir du corpus statique, et évaluation de
pertinence basée sur un LLM-as-judge (heuristique, pas une vérité absolue).
