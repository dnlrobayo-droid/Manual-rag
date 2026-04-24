"""
app/utils.py - Funciones auxiliares (embeddings nativos de ChromaDB, sin PyTorch)
"""
from __future__ import annotations
import os
from functools import lru_cache
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

def _get_config(key: str, default: str = "") -> str:
    val = os.getenv(key, "")
    if val:
        return val
    try:
        import streamlit as st
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return default

ANTHROPIC_API_KEY = _get_config("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = _get_config("ANTHROPIC_MODEL", "claude-3-haiku-20240307")
EMBEDDING_MODEL = _get_config("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHROMA_PERSIST_DIR = _get_config("CHROMA_PERSIST_DIR", "./chroma_db")
CHROMA_COLLECTION = _get_config("CHROMA_COLLECTION", "manuales_tecnicos")
MANUALS_DIR = _get_config("MANUALS_DIR", "./data/manuals")
CHUNK_SIZE = int(_get_config("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(_get_config("CHUNK_OVERLAP", "50"))
TOP_K = int(_get_config("TOP_K", "4"))

@lru_cache(maxsize=1)
def get_embedding_function():
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
    return DefaultEmbeddingFunction()

@lru_cache(maxsize=1)
def get_chroma_client():
    import chromadb
    from chromadb.config import Settings
    return chromadb.PersistentClient(
        path=CHROMA_PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False, allow_reset=True),
    )

def get_collection(create_if_missing: bool = True):
    client = get_chroma_client()
    ef = get_embedding_function()
    if create_if_missing:
        return client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
            embedding_function=ef,
        )
    return client.get_collection(name=CHROMA_COLLECTION, embedding_function=ef)

@lru_cache(maxsize=1)
def get_anthropic_client():
    from anthropic import Anthropic
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY no está configurada.")
    return Anthropic(api_key=ANTHROPIC_API_KEY)

def retrieve_chunks(query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    collection = get_collection(create_if_missing=True)
    if collection.count() == 0:
        return []
    results = collection.query(query_texts=[query], n_results=min(top_k, collection.count()))
    chunks = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]
    for text, meta, dist in zip(docs, metas, dists):
        chunks.append({
            "text": text,
            "source": meta.get("source", "desconocido"),
            "page": meta.get("page", "—"),
            "section": meta.get("section", "—"),
            "distance": round(float(dist), 4),
        })
    return chunks

SYSTEM_PROMPT = """Eres un técnico especialista en maquinaria industrial con acceso exclusivo a los manuales técnicos proporcionados en el contexto.

REGLAS ESTRICTAS:
1. Responde ÚNICAMENTE con información extraída de los fragmentos del manual proporcionados. Nunca inventes valores, referencias, códigos, torques, presiones ni procedimientos.
2. Si la información necesaria NO está en los fragmentos, responde literalmente:
   "No encontré información sobre [tema] en los manuales disponibles. Recomiendo consultar: [sugerencias basadas en términos cercanos encontrados en los fragmentos]"
3. Cita siempre el manual y la sección/página cuando estén disponibles en los metadatos.
4. No recomiendes acciones peligrosas sin mencionar las precauciones que aparezcan en el manual.
5. Responde en español técnico, claro y directo. Eres un colega ingeniero, no un asistente genérico.

FORMATO DE RESPUESTA OBLIGATORIO (usa Markdown):

**🔧 Problema identificado:**
Resumen en 1-2 frases de lo que pregunta el técnico.

**🔍 Causas probables:**
- Causa 1
- Causa 2

**🛠️ Procedimiento:**
1. Paso numerado
2. Paso numerado

**🧰 Herramientas necesarias:**
- Herramienta 1
- Herramienta 2

**⚠️ Precauciones de seguridad:**
- Advertencia 1
- Advertencia 2

**📖 Referencia del manual:**
Indica el archivo, sección y página citados en los fragmentos.

Si la consulta es solo informativa (ej. consulta de un código de error o un valor de torque), puedes omitir las secciones no aplicables y responder de forma más breve, pero mantén el estilo estructurado y la referencia al manual."""

def build_user_prompt(query: str, chunks: List[Dict[str, Any]], mode: str = "Consultas técnicas") -> str:
    if not chunks:
        return (f"El técnico consulta: {query}\n\nNo se recuperaron fragmentos de la base de conocimiento. "
                "Indica que no hay manuales indexados o que la consulta no tiene coincidencias.")
    context_blocks = []
    for i, ch in enumerate(chunks, 1):
        header = (f"[Fragmento {i}] Fuente: {ch['source']} | Página: {ch['page']} | "
                  f"Sección: {ch['section']} | Similitud (1-dist): {round(1 - ch['distance'], 3)}")
        context_blocks.append(f"{header}\n{ch['text']}")
    context = "\n\n---\n\n".join(context_blocks)
    mode_hint = {
        "Consultas técnicas": "El técnico necesita una respuesta técnica general.",
        "Procedimientos": "El técnico necesita un procedimiento paso a paso detallado.",
        "Búsqueda por código de error": "El técnico busca información sobre un código de error específico; prioriza la tabla de códigos.",
    }.get(mode, "El técnico necesita una respuesta técnica.")
    return (f"MODO DE CONSULTA: {mode}\nINSTRUCCIÓN ADICIONAL: {mode_hint}\n\n"
            f"=== FRAGMENTOS DEL MANUAL (contexto) ===\n\n{context}\n\n"
            f"=== CONSULTA DEL TÉCNICO ===\n{query}\n\n"
            f"Responde siguiendo el formato estructurado indicado en el system prompt. "
            f"Basa tu respuesta EXCLUSIVAMENTE en los fragmentos proporcionados.")

def ask_claude(query: str, chunks: List[Dict[str, Any]], mode: str = "Consultas técnicas", max_tokens: int = 1024) -> str:
    client = get_anthropic_client()
    user_prompt = build_user_prompt(query, chunks, mode=mode)
    message = client.messages.create(
        model=ANTHROPIC_MODEL, max_tokens=max_tokens, system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    parts = []
    for block in message.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "\n".join(parts).strip()

def format_chunk_for_display(chunk: Dict[str, Any], index: int) -> str:
    similarity_pct = round((1 - chunk["distance"]) * 100, 1)
    return (f"**Fragmento {index}** — similitud `{similarity_pct}%`  \n"
            f"📄 Fuente: `{chunk['source']}`  \n"
            f"📍 Página: `{chunk['page']}` · Sección: `{chunk['section']}`\n\n"
            f"> {chunk['text'][:800]}{'...' if len(chunk['text']) > 800 else ''}")

def get_index_stats() -> Dict[str, Any]:
    try:
        collection = get_collection(create_if_missing=True)
        count = collection.count()
        sources = set()
        if count > 0:
            data = collection.get(limit=min(count, 1000), include=["metadatas"])
            for meta in data.get("metadatas", []):
                if meta and "source" in meta:
                    sources.add(meta["source"])
        return {"total_chunks": count, "total_sources": len(sources), "sources": sorted(sources)}
    except Exception as e:
        return {"total_chunks": 0, "total_sources": 0, "sources": [], "error": str(e)}
