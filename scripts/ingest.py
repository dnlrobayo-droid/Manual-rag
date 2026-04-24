"""
scripts/ingest.py
=================
Procesa todos los manuales (PDF y TXT) de `data/manuals/`, los divide en
chunks jerárquicos, genera embeddings locales con SentenceTransformer y los
almacena en ChromaDB con persistencia en `./chroma_db`.

Uso:
    python scripts/ingest.py                # Ingesta incremental (añade)
    python scripts/ingest.py --reset        # Borra la colección y reindexa
    python scripts/ingest.py --dry-run      # Solo muestra qué se procesaría

Separadores jerárquicos: intenta dividir por secciones Markdown (##, ###),
luego por párrafos, líneas y finalmente caracteres.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import traceback
from pathlib import Path
from typing import List, Dict, Any

# Permitir importar desde app/
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.utils import (  # noqa: E402
    MANUALS_DIR, CHUNK_SIZE, CHUNK_OVERLAP, CHROMA_COLLECTION,
    get_chroma_client, get_collection,
)


# =========================================================================
# Carga de documentos (PDF y TXT)
# =========================================================================

def load_pdf(path: Path) -> List[Dict[str, Any]]:
    """Carga un PDF y devuelve una lista de dicts con texto y metadatos por página."""
    try:
        from langchain_community.document_loaders import PyPDFLoader
    except ImportError:
        print("[ERROR] langchain-community o pypdf no están instalados.")
        raise

    docs: List[Dict[str, Any]] = []
    try:
        loader = PyPDFLoader(str(path))
        pages = loader.load()
        for p in pages:
            docs.append({
                "text": p.page_content,
                "source": path.name,
                "page": p.metadata.get("page", "—"),
            })
    except Exception as e:
        print(f"[ADVERTENCIA] No se pudo leer '{path.name}': {e}")
    return docs


def load_txt(path: Path) -> List[Dict[str, Any]]:
    """Carga un archivo de texto como un único 'documento lógico'."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        print(f"[ADVERTENCIA] No se pudo leer '{path.name}': {e}")
        return []
    return [{"text": text, "source": path.name, "page": "—"}]


def load_all_documents(manuals_dir: Path) -> List[Dict[str, Any]]:
    """Carga todos los PDF/TXT de la carpeta de manuales."""
    if not manuals_dir.exists():
        print(f"[ERROR] La carpeta '{manuals_dir}' no existe.")
        return []

    pdf_files = sorted(manuals_dir.glob("*.pdf"))
    txt_files = sorted(manuals_dir.glob("*.txt"))
    all_files = pdf_files + txt_files

    if not all_files:
        print(f"[ADVERTENCIA] No se encontraron archivos .pdf ni .txt en '{manuals_dir}'.")
        return []

    print(f"📚 Archivos encontrados: {len(pdf_files)} PDF + {len(txt_files)} TXT = {len(all_files)} total")

    all_docs: List[Dict[str, Any]] = []
    for f in tqdm(all_files, desc="Cargando archivos", unit="archivo"):
        if f.suffix.lower() == ".pdf":
            all_docs.extend(load_pdf(f))
        else:
            all_docs.extend(load_txt(f))

    return all_docs


# =========================================================================
# Chunking jerárquico
# =========================================================================

def build_splitter() -> RecursiveCharacterTextSplitter:
    """
    Splitter que respeta la estructura del manual.

    Nota: `chunk_size` se mide en caracteres para este splitter; configuramos
    un valor de caracteres aproximadamente equivalente a CHUNK_SIZE tokens
    (≈4 caracteres por token como heurística).
    """
    approx_char_size = CHUNK_SIZE * 4
    approx_char_overlap = CHUNK_OVERLAP * 4

    return RecursiveCharacterTextSplitter(
        chunk_size=approx_char_size,
        chunk_overlap=approx_char_overlap,
        length_function=len,
        separators=[
            "\n## ",    # Encabezado H2 (sección principal)
            "\n### ",   # Encabezado H3 (subsección)
            "\n#### ",  # Encabezado H4
            "\n\n",     # Párrafos
            "\n",       # Líneas
            ". ",       # Frases
            " ",        # Palabras
            "",
        ],
        keep_separator=True,
    )


def detect_section(chunk_text: str) -> str:
    """Intenta extraer un título de sección del inicio del chunk."""
    for line in chunk_text.splitlines()[:5]:
        s = line.strip()
        if s.startswith(("## ", "### ", "#### ")):
            return s.lstrip("#").strip()[:120]
    return "—"


def chunk_documents(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    splitter = build_splitter()
    all_chunks: List[Dict[str, Any]] = []

    for doc in tqdm(docs, desc="Dividiendo en chunks", unit="doc"):
        pieces = splitter.split_text(doc["text"])
        for piece in pieces:
            piece_clean = piece.strip()
            if len(piece_clean) < 30:
                # Descartamos fragmentos muy cortos (ruido)
                continue
            all_chunks.append({
                "text": piece_clean,
                "source": doc["source"],
                "page": str(doc.get("page", "—")),
                "section": detect_section(piece_clean),
            })
    return all_chunks


# =========================================================================
# Generación de embeddings e inserción en ChromaDB
# =========================================================================

def compute_chunk_id(chunk: Dict[str, Any]) -> str:
    """ID determinista basado en fuente + página + hash del texto."""
    h = hashlib.sha1(chunk["text"].encode("utf-8")).hexdigest()[:12]
    return f"{chunk['source']}::p{chunk['page']}::{h}"


def index_chunks(chunks, reset=False):
    client = get_chroma_client()
    if reset:
        try:
            client.delete_collection(CHROMA_COLLECTION)
            print(f"Colección '{CHROMA_COLLECTION}' eliminada.")
        except Exception:
            pass
    collection = get_collection(create_if_missing=True)
    texts = [c["text"] for c in chunks]
    ids = [compute_chunk_id(c) for c in chunks]
    metadatas = [{"source": c["source"], "page": c["page"], "section": c["section"]} for c in chunks]
    BATCH = 256
    print(f"Generando embeddings e insertando en ChromaDB (lotes de {BATCH})...")
    for i in tqdm(range(0, len(ids), BATCH), desc="Upsert", unit="lote"):
        collection.upsert(
            ids=ids[i:i + BATCH],
            documents=texts[i:i + BATCH],
            metadatas=metadatas[i:i + BATCH],
        )


# =========================================================================
# Main
# =========================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="Ingesta de manuales técnicos en ChromaDB")
    parser.add_argument("--reset", action="store_true",
                        help="Borra la colección antes de indexar (reindexado completo).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Solo muestra qué se procesaría, sin escribir en ChromaDB.")
    args = parser.parse_args()

    print("=" * 60)
    print("  Manual RAG — Ingesta de documentos")
    print("=" * 60)
    print(f"📁 Carpeta de manuales: {MANUALS_DIR}")
    print(f"🧩 Chunk size: {CHUNK_SIZE} tokens (~{CHUNK_SIZE*4} chars), overlap: {CHUNK_OVERLAP}")
    print(f"💽 ChromaDB dir: {os.getenv('CHROMA_PERSIST_DIR', './chroma_db')}")
    print(f"🏷️  Colección: {CHROMA_COLLECTION}")
    print("-" * 60)

    manuals_dir = Path(MANUALS_DIR)

    try:
        # 1. Cargar documentos
        docs = load_all_documents(manuals_dir)
        if not docs:
            print("\n[RESULTADO] Sin documentos. Añade archivos a data/manuals/ y reintenta.")
            return 1

        # 2. Chunking
        chunks = chunk_documents(docs)
        print(f"\n✅ {len(docs)} documentos → {len(chunks)} chunks")

        if args.dry_run:
            print("\n[DRY RUN] No se escribirá en ChromaDB.")
            print("Ejemplo de los primeros 3 chunks:")
            for i, c in enumerate(chunks[:3], 1):
                print(f"\n--- Chunk {i} ---")
                print(f"  Fuente: {c['source']} (pág. {c['page']})")
                print(f"  Sección: {c['section']}")
                print(f"  Texto: {c['text'][:200]}...")
            return 0

        # 3. Indexar
        index_chunks(chunks, reset=args.reset)

        # 4. Resumen final
        final = get_collection(create_if_missing=True)
        print("\n" + "=" * 60)
        print(f"  ✅ INGESTA COMPLETADA")
        print("=" * 60)
        print(f"  📊 Chunks totales en la colección: {final.count()}")
        print(f"  🗂️  Fuentes procesadas: {len(set(c['source'] for c in chunks))}")
        for src in sorted(set(c['source'] for c in chunks)):
            n = sum(1 for c in chunks if c['source'] == src)
            print(f"     • {src}: {n} chunks")
        print("=" * 60)
        return 0

    except KeyboardInterrupt:
        print("\n[CANCELADO] Interrumpido por el usuario.")
        return 130
    except Exception as e:
        print(f"\n[ERROR] {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
