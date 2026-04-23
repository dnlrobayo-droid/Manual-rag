"""
app/utils.py
============
Funciones auxiliares compartidas por la app Streamlit y los scripts.

Contiene:
- Carga perezosa (cached) del modelo de embeddings.
- Carga perezosa del cliente de ChromaDB y la colección.
- Carga perezosa del cliente de Anthropic.
- Función de recuperación de chunks relevantes.
- Construcción del system prompt y del user prompt RAG.
- Formateo de respuestas y de fragmentos para UI.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv

# Cargar variables de entorno una sola vez
load_dotenv()

# =========================================================================
# Configuración centralizada
# =========================================================================

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "manuales_tecnicos")
MANUALS_DIR = os.getenv("MANUALS_DIR", "./data/manuals")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
TOP_K = int(os.getenv("TOP_K", "4"))


# =========================================================================
# Carga perezosa de recursos pesados (LRU cache)
# =========================================================================

@lru_cache(maxsize=1)
def get_embedding_model():
    """Carga y cachea el modelo SentenceTransformer local."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_chroma_client():
    """Crea y cachea el cliente persistente de ChromaDB."""
    import chromadb
    from chromadb.config import Settings

    return chromadb.PersistentClient(
        path=CHROMA_PERSIST_DIR,
        settings=Settings(anonymized_telemetry=False, allow_reset=True),
    )


def get_collection(create_if_missing: bool = True):
    """Devuelve la colección de ChromaDB, creándola si no existe."""
    client = get_chroma_client()
    if create_if_missing:
        return client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
    return client.get_collection(name=CHROMA_COLLECTION)


@lru_cache(maxsize=1)
def get_anthropic_client():
    """Instancia y cachea el cliente de Anthropic."""
    from anthropic import Anthropic
    if not ANTHROPIC_API_KEY:
        raise ValueError(
            "ANTHROPIC_API_KEY no está configurada. "
            "Copia .env.example como .env y añade tu clave."
        )
    return Anthropic(api_key=ANTHROPIC_API_KEY)


# =========================================================================
# Recuperación de chunks (R de RAG)
# =========================================================================

def embed_query(query: str) -> List[float]:
    """Calcula el embedding de una consulta de usuario."""
    model = get_embedding_model()
    return model.encode(query, convert_to_numpy=True).tolist()


def retrieve_chunks(query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    """
    Recupera los top_k chunks más similares a la consulta.

    Devuelve una lista de dicts con:
        - text: el contenido del chunk
        - source: nombre del archivo origen
        - page: página (si procede)
        - section: sección detectada (si procede)
        - distance: distancia coseno (menor = más similar)
    """
    collection = get_collection(create_if_missing=True)

    if collection.count() == 0:
        return []

    query_embedding = embed_query(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
    )

    chunks: List[Dict[str, Any]] = []
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


# =========================================================================
# Prompting para Claude
# =========================================================================

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
- ...

**🛠️ Procedimiento:**
1. Paso numerado
2. Paso numerado
3. ...

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
    """Construye el mensaje de usuario con el contexto recuperado."""
    if not chunks:
        return (
            f"El técnico consulta: {query}\n\n"
            "No se recuperaron fragmentos de la base de conocimiento. "
            "Indica que no hay manuales indexados o que la consulta no tiene coincidencias."
        )

    context_blocks = []
    for i, ch in enumerate(chunks, 1):
        header = (
            f"[Fragmento {i}] Fuente: {ch['source']} | "
            f"Página: {ch['page']} | Sección: {ch['section']} | "
            f"Similitud (1-dist): {round(1 - ch['distance'], 3)}"
        )
        context_blocks.append(f"{header}\n{ch['text']}")

    context = "\n\n---\n\n".join(context_blocks)

    mode_hint = {
        "Consultas técnicas": "El técnico necesita una respuesta técnica general.",
        "Procedimientos": "El técnico necesita un procedimiento paso a paso detallado.",
        "Búsqueda por código de error": "El técnico busca información sobre un código de error específico; prioriza la tabla de códigos.",
    }.get(mode, "El técnico necesita una respuesta técnica.")

    return (
        f"MODO DE CONSULTA: {mode}\n"
        f"INSTRUCCIÓN ADICIONAL: {mode_hint}\n\n"
        f"=== FRAGMENTOS DEL MANUAL (contexto) ===\n\n"
        f"{context}\n\n"
        f"=== CONSULTA DEL TÉCNICO ===\n"
        f"{query}\n\n"
        f"Responde siguiendo el formato estructurado indicado en el system prompt. "
        f"Basa tu respuesta EXCLUSIVAMENTE en los fragmentos proporcionados."
    )


def ask_claude(query: str, chunks: List[Dict[str, Any]], mode: str = "Consultas técnicas",
               max_tokens: int = 1024) -> str:
    """Envía la consulta a Claude y devuelve la respuesta en texto."""
    client = get_anthropic_client()
    user_prompt = build_user_prompt(query, chunks, mode=mode)

    message = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    # La respuesta viene como lista de content blocks
    parts = []
    for block in message.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "\n".join(parts).strip()


# =========================================================================
# Utilidades de formato para la UI
# =========================================================================

def format_chunk_for_display(chunk: Dict[str, Any], index: int) -> str:
    """Formatea un chunk recuperado para mostrarlo en el expander de Streamlit."""
    similarity_pct = round((1 - chunk["distance"]) * 100, 1)
    return (
        f"**Fragmento {index}** — similitud `{similarity_pct}%`  \n"
        f"📄 Fuente: `{chunk['source']}`  \n"
        f"📍 Página: `{chunk['page']}` · Sección: `{chunk['section']}`\n\n"
        f"> {chunk['text'][:800]}{'...' if len(chunk['text']) > 800 else ''}"
    )


def get_index_stats() -> Dict[str, Any]:
    """Devuelve estadísticas actuales del índice para el sidebar."""
    try:
        collection = get_collection(create_if_missing=True)
        count = collection.count()

        # Fuentes únicas
        sources = set()
        if count > 0:
            # Muestreo eficiente (hasta 1000 chunks) para listar fuentes únicas
            data = collection.get(limit=min(count, 1000), include=["metadatas"])
            for meta in data.get("metadatas", []):
                if meta and "source" in meta:
                    sources.add(meta["source"])

        return {
            "total_chunks": count,
            "total_sources": len(sources),
            "sources": sorted(sources),
        }
    except Exception as e:
        return {"total_chunks": 0, "total_sources": 0, "sources": [], "error": str(e)}
