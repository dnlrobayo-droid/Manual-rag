"""
scripts/query.py
================
Prueba rápida por terminal del sistema RAG.
Permite consultar el índice sin interfaz gráfica.

Uso:
    python scripts/query.py "¿qué significa el código de error E05?"
    python scripts/query.py                # Modo interactivo (REPL)
    python scripts/query.py --top-k 6 "..." # Más fragmentos recuperados
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Permitir importar desde app/
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.utils import retrieve_chunks, ask_claude, get_index_stats  # noqa: E402


def print_header():
    print("=" * 60)
    print("  Manual RAG — Consulta por terminal")
    print("=" * 60)
    stats = get_index_stats()
    print(f"  📊 Chunks indexados: {stats['total_chunks']}")
    print(f"  🗂️  Fuentes: {stats['total_sources']}")
    if stats["total_chunks"] == 0:
        print("\n  ⚠️  Base de conocimiento VACÍA.")
        print("     Ejecuta primero: python scripts/ingest.py")
    print("=" * 60)


def run_single_query(query: str, top_k: int, show_chunks: bool, mode: str) -> int:
    print(f"\n🔎 Consulta: {query}")
    print(f"📂 Modo: {mode} | top_k: {top_k}\n")

    chunks = retrieve_chunks(query, top_k=top_k)

    if not chunks:
        print("⚠️  No hay fragmentos indexados o no hay coincidencias.")
        print("   Ejecuta 'python scripts/ingest.py' si aún no has indexado.")
        return 1

    if show_chunks:
        print("=" * 60)
        print("  📄 FRAGMENTOS RECUPERADOS")
        print("=" * 60)
        for i, ch in enumerate(chunks, 1):
            sim = round((1 - ch["distance"]) * 100, 1)
            print(f"\n[{i}] similitud {sim}% | {ch['source']} · pág {ch['page']} · {ch['section']}")
            print("-" * 60)
            snippet = ch["text"][:500]
            print(snippet + ("..." if len(ch["text"]) > 500 else ""))
        print()

    print("=" * 60)
    print("  🤖 RESPUESTA DE CLAUDE")
    print("=" * 60)
    try:
        answer = ask_claude(query, chunks, mode=mode)
    except Exception as e:
        print(f"[ERROR] Al llamar a la API de Anthropic: {e}")
        return 1
    print(answer)
    print()
    return 0


def run_repl(top_k: int, show_chunks: bool, mode: str) -> int:
    print("\n💡 Modo interactivo. Escribe 'salir', 'exit' o Ctrl+C para terminar.\n")
    while True:
        try:
            query = input("▶ Consulta: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Hasta luego.")
            return 0
        if not query:
            continue
        if query.lower() in {"salir", "exit", "quit"}:
            print("👋 Hasta luego.")
            return 0
        run_single_query(query, top_k=top_k, show_chunks=show_chunks, mode=mode)


def main() -> int:
    parser = argparse.ArgumentParser(description="Consulta RAG por terminal.")
    parser.add_argument("query", nargs="?", help="Consulta. Si se omite, entra en modo interactivo.")
    parser.add_argument("--top-k", type=int, default=4, help="Nº de fragmentos a recuperar (def. 4).")
    parser.add_argument("--no-chunks", action="store_true", help="No mostrar los fragmentos recuperados.")
    parser.add_argument(
        "--mode",
        default="Consultas técnicas",
        choices=["Consultas técnicas", "Procedimientos", "Búsqueda por código de error"],
        help="Modo de consulta.",
    )
    args = parser.parse_args()

    print_header()
    show_chunks = not args.no_chunks

    if args.query:
        return run_single_query(args.query, args.top_k, show_chunks, args.mode)
    return run_repl(args.top_k, show_chunks, args.mode)


if __name__ == "__main__":
    sys.exit(main())
