#!/usr/bin/env python3
"""
Ponto de entrada CLI da PoC de RAG.

Uso:
  # Indexar um PDF e fazer perguntas interativamente
  python main.py --pdf caminho/para/documento.pdf

  # Forçar reindexação (quando trocar o PDF)
  python main.py --pdf documento.pdf --rebuild

  # Modo verbose: mostra os chunks recuperados antes da resposta
  python main.py --pdf documento.pdf --verbose
"""

import argparse
import sys
from pathlib import Path

# permite rodar main.py da raiz do projeto
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rag import ask, build_vector_store

BANNER = """
╔══════════════════════════════════════════════╗
║          RAG PoC — PDF Question Answering    ║
║  Retrieval Augmented Generation from scratch ║
╚══════════════════════════════════════════════╝
"""


def main():
    parser = argparse.ArgumentParser(description="RAG PoC — pergunte sobre qualquer PDF")
    parser.add_argument("--pdf", required=True, help="Caminho para o arquivo PDF")
    parser.add_argument("--rebuild", action="store_true", help="Força reindexação do PDF")
    parser.add_argument("--verbose", action="store_true", help="Mostra chunks recuperados")
    args = parser.parse_args()

    if not Path(args.pdf).exists():
        print(f"[ERRO] Arquivo não encontrado: {args.pdf}")
        sys.exit(1)

    print(BANNER)

    # ── Ingestion ──────────────────────────────────────────────────────────────
    print("[ INGESTION ]")
    vector_store = build_vector_store(args.pdf, force_rebuild=args.rebuild)
    n_chunks = len(vector_store["chunks"])
    print(f"✓ {n_chunks} chunks indexados\n")

    # ── Loop de perguntas ──────────────────────────────────────────────────────
    print("[ PRONTO ] Digite sua pergunta (ou 'sair' para encerrar)\n")

    while True:
        try:
            query = input("Pergunta: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nEncerrando.")
            break

        if not query:
            continue

        if query.lower() in {"sair", "exit", "quit"}:
            print("Encerrando.")
            break

        print("\n[ RETRIEVAL + GENERATION ]")
        answer = ask(query, vector_store, verbose=args.verbose)
        print(f"\nResposta:\n{answer}\n")
        print("─" * 60 + "\n")


if __name__ == "__main__":
    main()
