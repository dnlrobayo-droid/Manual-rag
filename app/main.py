"""
app/main.py
===========
Interfaz Streamlit de Manual RAG — Asistente Técnico.

Ejecutar:
    streamlit run app/main.py
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

# Permitir importar desde la raíz cuando se ejecuta `streamlit run app/main.py`
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.utils import (  # noqa: E402
    retrieve_chunks,
    ask_claude,
    get_index_stats,
    format_chunk_for_display,
    get_embedding_model,
    get_chroma_client,
    ANTHROPIC_MODEL,
    EMBEDDING_MODEL,
    TOP_K,
)


# =========================================================================
# Configuración de página
# =========================================================================

st.set_page_config(
    page_title="Manual RAG — Asistente Técnico",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Tema oscuro industrial con CSS personalizado
st.markdown(
    """
    <style>
        /* Fondo principal oscuro industrial */
        .stApp {
            background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
        }
        /* Títulos con acento amarillo "industrial" */
        h1, h2, h3 {
            color: #f0f6fc !important;
            letter-spacing: 0.3px;
        }
        h1 { border-bottom: 2px solid #f0b429; padding-bottom: 8px; }
        /* Caja de texto */
        .stTextArea textarea {
            background-color: #0d1117 !important;
            color: #e6edf3 !important;
            border: 1px solid #30363d !important;
            font-family: 'Fira Code', monospace;
        }
        /* Botón primario */
        .stButton>button[kind="primary"] {
            background-color: #f0b429 !important;
            color: #0d1117 !important;
            font-weight: 700 !important;
            border: none !important;
        }
        .stButton>button[kind="primary"]:hover {
            background-color: #fbbf24 !important;
        }
        /* Tarjetas de respuesta */
        .answer-card {
            background-color: #161b22;
            border-left: 4px solid #f0b429;
            border-radius: 8px;
            padding: 20px;
            margin: 16px 0;
            color: #e6edf3;
        }
        /* Chip de estado */
        .status-chip {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: 600;
            margin-right: 6px;
        }
        .status-ok   { background: #1f6feb33; color: #58a6ff; border: 1px solid #1f6feb; }
        .status-warn { background: #9e6a0333; color: #f0b429; border: 1px solid #9e6a03; }
        .status-err  { background: #da363333; color: #f85149; border: 1px solid #da3633; }
        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #010409;
            border-right: 1px solid #30363d;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================================
# Estado de sesión
# =========================================================================

if "history" not in st.session_state:
    st.session_state.history = []  # [{query, answer, chunks, timestamp, mode}]

if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = datetime.now()


# =========================================================================
# SIDEBAR
# =========================================================================

with st.sidebar:
    st.markdown("# 🔧 Manual RAG")
    st.caption("Asistente Técnico para Maquinaria Industrial")
    st.divider()

    # Selector de modo
    st.markdown("### 🎯 Modo de consulta")
    mode = st.radio(
        label="Selecciona el tipo de consulta",
        options=["Consultas técnicas", "Procedimientos", "Búsqueda por código de error"],
        index=0,
        label_visibility="collapsed",
    )

    top_k = st.slider("Fragmentos a recuperar (top-k)", min_value=2, max_value=10, value=TOP_K)

    st.divider()

    # Información de la sesión / índice
    st.markdown("### 📊 Estado del índice")
    stats = get_index_stats()
    total_chunks = stats["total_chunks"]
    total_sources = stats["total_sources"]

    if total_chunks == 0:
        st.markdown(
            '<span class="status-chip status-err">⚠ Sin documentos</span>',
            unsafe_allow_html=True,
        )
        st.info(
            "La base de conocimiento está vacía.\n\n"
            "1. Coloca PDFs/TXT en `data/manuals/`\n"
            "2. Ejecuta en terminal:\n\n"
            "`python scripts/ingest.py`"
        )
    else:
        st.markdown(
            '<span class="status-chip status-ok">✓ Operativo</span>',
            unsafe_allow_html=True,
        )
        col1, col2 = st.columns(2)
        col1.metric("Chunks", f"{total_chunks:,}")
        col2.metric("Manuales", total_sources)

        with st.expander(f"📚 Manuales indexados ({total_sources})"):
            for src in stats["sources"]:
                st.markdown(f"- `{src}`")

    st.caption(
        f"🕒 Última sincronización: "
        f"{st.session_state.last_refresh.strftime('%H:%M:%S')}"
    )

    if st.button("🔄 Recargar base de conocimiento", use_container_width=True):
        # Limpiar cachés de recursos (cliente Chroma, embeddings)
        get_chroma_client.cache_clear()
        get_embedding_model.cache_clear()
        st.session_state.last_refresh = datetime.now()
        st.rerun()

    st.divider()

    # Configuración activa
    with st.expander("⚙️ Configuración activa"):
        st.markdown(f"**LLM:** `{ANTHROPIC_MODEL}`")
        st.markdown(f"**Embeddings:** `{EMBEDDING_MODEL}`")
        st.markdown(f"**Top-K:** `{top_k}`")

    # Historial
    st.divider()
    st.markdown("### 🗂️ Historial de la sesión")
    if not st.session_state.history:
        st.caption("Aún no hay consultas.")
    else:
        for i, item in enumerate(reversed(st.session_state.history[-10:]), 1):
            ts = item["timestamp"].strftime("%H:%M")
            preview = item["query"][:40] + ("..." if len(item["query"]) > 40 else "")
            st.caption(f"`{ts}` · {preview}")

        if st.button("🗑️ Limpiar historial", use_container_width=True):
            st.session_state.history = []
            st.rerun()


# =========================================================================
# ÁREA CENTRAL
# =========================================================================

st.title("🔧 Manual RAG — Asistente Técnico")
st.markdown(
    "Consulta tus manuales técnicos en lenguaje natural. "
    "Respuestas basadas **exclusivamente** en la documentación indexada."
)

# Validaciones previas
import os
if not os.getenv("ANTHROPIC_API_KEY") and not st.secrets.get("ANTHROPIC_API_KEY", None) if hasattr(st, "secrets") else True:
    # Intento suave: Streamlit Cloud usa st.secrets; local usa .env
    pass

# Placeholder de ejemplos según el modo
mode_placeholders = {
    "Consultas técnicas": "Ej: ¿Cuál es el torque correcto para los pernos del cabezal del compresor IC-750?",
    "Procedimientos": "Ej: Explícame paso a paso cómo purgar el sistema hidráulico tras un cambio de fluido.",
    "Búsqueda por código de error": "Ej: ¿Qué significa el código de error E05 y cómo lo soluciono?",
}

st.markdown("### 💬 Consulta")
query = st.text_area(
    label="Describe tu consulta técnica",
    placeholder=mode_placeholders[mode],
    height=120,
    label_visibility="collapsed",
)

col_btn1, col_btn2, _ = st.columns([1, 1, 3])
with col_btn1:
    consult_btn = st.button("🔎 Consultar Manual", type="primary", use_container_width=True)
with col_btn2:
    clear_btn = st.button("🧹 Limpiar", use_container_width=True)

if clear_btn:
    st.rerun()


# =========================================================================
# PROCESAMIENTO DE LA CONSULTA
# =========================================================================

def process_query(q: str, mode: str, top_k: int) -> None:
    """Ejecuta la pipeline RAG y renderiza la respuesta."""
    if total_chunks == 0:
        st.error(
            "⚠️ No hay manuales indexados. "
            "Ejecuta `python scripts/ingest.py` antes de consultar."
        )
        return

    if not q.strip():
        st.warning("Escribe una consulta antes de pulsar el botón.")
        return

    # 1. Recuperar fragmentos
    with st.spinner("🔍 Buscando en los manuales..."):
        t0 = time.time()
        try:
            chunks = retrieve_chunks(q, top_k=top_k)
        except Exception as e:
            st.error(f"Error al recuperar fragmentos: {e}")
            return
        t_retrieve = time.time() - t0

    if not chunks:
        st.warning("No se encontraron fragmentos relevantes para esta consulta.")
        return

    # 2. Generar respuesta con Claude
    with st.spinner("🤖 Consultando al especialista técnico (Claude)..."):
        t0 = time.time()
        try:
            answer = ask_claude(q, chunks, mode=mode)
        except Exception as e:
            st.error(
                f"Error al contactar con la API de Anthropic:\n\n`{e}`\n\n"
                "Verifica que `ANTHROPIC_API_KEY` esté configurada correctamente."
            )
            return
        t_claude = time.time() - t0

    # 3. Renderizar
    st.markdown("### 🤖 Respuesta")
    st.markdown(f'<div class="answer-card">{answer}</div>', unsafe_allow_html=True)

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("⏱️ Recuperación", f"{t_retrieve:.2f} s")
    col_m2.metric("⏱️ Generación", f"{t_claude:.2f} s")
    col_m3.metric("📄 Fragmentos", len(chunks))

    with st.expander(f"📄 Ver fragmentos del manual consultados ({len(chunks)})"):
        for i, ch in enumerate(chunks, 1):
            st.markdown(format_chunk_for_display(ch, i))
            st.divider()

    # 4. Guardar en historial
    st.session_state.history.append({
        "query": q,
        "answer": answer,
        "chunks": chunks,
        "timestamp": datetime.now(),
        "mode": mode,
    })


if consult_btn:
    process_query(query, mode, top_k)


# =========================================================================
# HISTORIAL RENDERIZADO (últimas consultas)
# =========================================================================

if st.session_state.history and not consult_btn:
    st.divider()
    with st.expander(f"📜 Últimas {min(3, len(st.session_state.history))} consultas de esta sesión"):
        for item in reversed(st.session_state.history[-3:]):
            st.markdown(f"**🕒 {item['timestamp'].strftime('%H:%M:%S')} — Modo: {item['mode']}**")
            st.markdown(f"*Consulta:* {item['query']}")
            st.markdown(f'<div class="answer-card">{item["answer"]}</div>', unsafe_allow_html=True)
            st.divider()


# =========================================================================
# FOOTER
# =========================================================================

st.divider()
st.caption(
    "🔧 Manual RAG v1.0 · Powered by **Claude 3 Haiku** + **ChromaDB** + **SentenceTransformer** · "
    "Desarrollado por un técnico mecánico con maestría en IA."
)
