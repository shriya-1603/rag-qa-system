import os

import streamlit as st

from app.ingestion import ingest_file, SUPPORTED_EXTENSIONS
from app.vectorstore import add_to_vectorstore, load_vectorstore, vectorstore_exists, clear_vectorstore
from app.qa_chain import build_qa_chain
from config import RAW_DIR

st.set_page_config(page_title="RAG Doc Q&A", page_icon="🗞️", layout="wide")

# ── Dark-mode state ───────────────────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

dark = st.session_state.dark_mode
BG     = "#0a0a0a" if dark else "#ffffff"
FG     = "#ffffff" if dark else "#000000"
BG2    = "#1a1a1a" if dark else "#f2f2f2"
BG3    = "#111111" if dark else "#fafafa"
DOT    = "#333333" if dark else "#cccccc"
MUTED  = "#999999" if dark else "#555555"

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Permanent+Marker&family=Comic+Neue:ital,wght@0,400;0,700;1,700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&display=swap');

/* ── Override Streamlit CSS variables (kills the red focus ring) ── */
:root {{
    --primary-color: {FG} !important;
    --background-color: {BG} !important;
    --secondary-background-color: {BG2} !important;
    --text-color: {FG} !important;
    --font: 'Comic Neue', cursive !important;
}}

/* ── Nuclear text-color reset ── */
* {{
    color: {FG} !important;
}}

/* ── Inverted-text helper — class beats * even with !important ── */
.inv {{
    color: {BG} !important;
    font-family: 'Permanent Marker', cursive !important;
}}

/* ── Restore Material icon font (prevents "keyboard_double_arrow_right" text) ── */
.material-symbols-rounded,
.material-symbols-outlined,
.material-icons-round,
.material-icons,
span[class*="material-symbol"],
span[class*="material-icon"] {{
    font-family: 'Material Symbols Rounded', 'Material Icons Round', 'Material Icons', sans-serif !important;
    color: {FG} !important;
    font-size: 20px !important;
    -webkit-font-feature-settings: 'liga' !important;
    font-feature-settings: 'liga' !important;
}}

/* ── Comic font on content ── */
/* NOTE: 'button' intentionally omitted — icon-only buttons (chat send,
   sidebar toggle, etc.) use Material Symbols ligatures that break if
   the font-family is overridden. Buttons get Comic Neue via their own
   more-specific rules below. */
html, body, .stApp,
p, div, label, li, td, th, a, small, strong, em, input, textarea, h1, h2, h3, h4 {{
    font-family: 'Comic Neue', 'Comic Sans MS', cursive !important;
}}

/* ── App background ── */
html, body, .stApp {{
    background-color: {BG} !important;
    background-image: radial-gradient({DOT} 1px, transparent 1px) !important;
    background-size: 22px 22px !important;
}}

/* ── Streamlit top header / toolbar ── */
header[data-testid="stHeader"] {{
    background-color: {BG} !important;
    border-bottom: 3px solid {FG} !important;
}}
/* Force text colour on ALL header children — no fill override so the
   Lottie cycling animation keeps its own colours. background-color set
   to transparent so no accidental dark boxes appear. */
header[data-testid="stHeader"] * {{
    color: {FG} !important;
    background-color: transparent !important;
}}
/* EXCEPTION: restore natural rendering inside the status widget so the
   cycling-man Lottie animation is not affected. This selector has the
   same specificity as the rule above but comes later, so it wins. */
header [data-testid="stStatusWidget"] *,
header [data-testid="stStatusWidget"] svg * {{
    color: revert !important;
    background-color: revert !important;
}}

/* ALL header buttons: no borders by default so icon-only buttons don't
   turn into solid black squares */
header[data-testid="stHeader"] button {{
    background-color: {BG} !important;
    border: none !important;
    box-shadow: none !important;
    padding: 4px 8px !important;
    margin: 0 2px !important;
}}

/* Cartoon border only for text-label buttons (Stop / Deploy).
   :has(> span:not([class*="material"])) matches buttons whose direct
   child span is NOT a Material icon, i.e. contains real text. */
header[data-testid="stHeader"] button:not([aria-haspopup]):has(> span:not([class*="material"])) {{
    background-color: {BG} !important;
    border: 2px solid {FG} !important;
    border-radius: 0 !important;
    box-shadow: 3px 3px 0 {FG} !important;
    font-family: 'Comic Neue', cursive !important;
    font-weight: 900 !important;
    font-size: 0.85rem !important;
    text-transform: uppercase !important;
    padding: 2px 10px !important;
}}
header[data-testid="stHeader"] button:not([aria-haspopup]):has(> span:not([class*="material"])):hover {{
    background-color: {FG} !important;
    color: {BG} !important;
}}

/* ── Status widget (running indicator / cycling man animation) ──
   We must NOT touch its SVG fills — the Lottie animation has its own
   colour data. Just ensure the container is transparent. */
[data-testid="stStatusWidget"],
[data-testid="stStatusWidget"] > div,
[data-testid="stStatusWidget"] button {{
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
}}
/* Reset colour inheritance so currentColor doesn't bleed into Lottie */
[data-testid="stStatusWidget"] * {{
    color: revert !important;
}}

/* ── 3-dots / main menu button ──
   Target every possible test-id and aria attribute Streamlit may use */
[data-testid="stMainMenu"] > button,
[data-testid="stMainMenuButton"],
[data-testid="stMainMenuButton"] > button,
header button[aria-haspopup],
header button[aria-haspopup="true"],
header button[aria-haspopup="menu"],
header button[aria-expanded] {{
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 4px !important;
}}

/* ── Main menu popup (BaseWeb popover) ── */
[data-baseweb="popover"],
[data-baseweb="popover"] > div,
[data-baseweb="popover"] > div > div {{
    background-color: {BG} !important;
    border: 3px solid {FG} !important;
    box-shadow: 6px 6px 0 {FG} !important;
    border-radius: 0 !important;
}}
[data-baseweb="menu"],
[data-baseweb="menu"] > ul,
[role="listbox"] {{
    background-color: {BG} !important;
    border-radius: 0 !important;
}}
[data-baseweb="menu"] li,
[data-baseweb="menu"] [role="option"],
[data-baseweb="list-item"],
[role="listbox"] li,
[role="menuitem"] {{
    background-color: {BG} !important;
    color: {FG} !important;
    font-family: 'Comic Neue', cursive !important;
    font-weight: 700 !important;
    border-radius: 0 !important;
}}
[data-baseweb="menu"] li:hover,
[data-baseweb="menu"] [role="option"]:hover,
[data-baseweb="list-item"]:hover {{
    background-color: {FG} !important;
    color: {BG} !important;
}}
[data-baseweb="menu"] li:hover *,
[data-baseweb="list-item"]:hover * {{
    color: {BG} !important;
}}
[data-baseweb="divider"] {{
    background-color: {FG} !important;
    opacity: 0.3 !important;
}}

/* ── Sidebar collapse / expand button ── */
[data-testid="stSidebarCollapseButton"] button,
[data-testid="stSidebarCollapsedControl"] button {{
    background-color: {BG} !important;
    border: 2px solid {FG} !important;
    border-radius: 0 !important;
    box-shadow: 3px 3px 0 {FG} !important;
}}
[data-testid="stSidebarCollapseButton"] span,
[data-testid="stSidebarCollapsedControl"] span {{
    font-family: 'Material Symbols Rounded', 'Material Icons', sans-serif !important;
    color: {FG} !important;
    font-size: 18px !important;
}}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {{
    background-color: {BG2} !important;
    border-right: 5px solid {FG} !important;
}}
section[data-testid="stSidebar"] * {{
    color: {FG} !important;
    background-color: transparent !important;
}}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {{
    font-family: 'Permanent Marker', cursive !important;
    background-color: {FG} !important;
    color: {BG} !important;
    padding: 4px 12px !important;
    display: inline-block !important;
    box-shadow: 4px 4px 0 {MUTED} !important;
    margin-bottom: 14px !important;
}}
section[data-testid="stSidebar"] hr {{
    border-color: {FG} !important;
    border-width: 2px !important;
    opacity: 1 !important;
}}

/* ── File uploader ── */
[data-testid="stFileUploader"] {{
    border: 3px dashed {FG} !important;
    border-radius: 0 !important;
    background-color: {BG3} !important;
    padding: 8px !important;
}}
[data-testid="stFileUploaderDropzone"],
[data-testid="stFileUploaderDropzone"] > div {{
    background-color: {BG3} !important;
    border: none !important;
}}
[data-testid="stFileUploaderDropzone"] * {{
    color: {FG} !important;
}}
[data-testid="stFileUploaderDropzone"] svg,
[data-testid="stFileUploaderDropzone"] svg path {{
    fill: {FG} !important;
    color: {FG} !important;
}}
/* Browse files button */
[data-testid="stFileUploaderDropzone"] button,
[data-testid="stFileUploader"] [data-testid="stBaseButton-secondary"] {{
    background-color: {BG} !important;
    color: {FG} !important;
    border: 3px solid {FG} !important;
    border-radius: 0 !important;
    font-family: 'Comic Neue', cursive !important;
    font-weight: 900 !important;
    font-size: 0.88rem !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    box-shadow: 4px 4px 0 {FG} !important;
    padding: 5px 14px !important;
    transition: all 0.06s ease !important;
}}
[data-testid="stFileUploaderDropzone"] button:hover,
[data-testid="stFileUploader"] [data-testid="stBaseButton-secondary"]:hover {{
    background-color: {FG} !important;
    color: {BG} !important;
    box-shadow: 2px 2px 0 {FG} !important;
    transform: translate(2px, 2px) !important;
}}
[data-testid="stFileUploaderDropzone"] button *,
[data-testid="stFileUploader"] [data-testid="stBaseButton-secondary"] * {{
    color: inherit !important;
}}

/* ── Action buttons ── */
.stButton > button {{
    background-color: {BG} !important;
    color: {FG} !important;
    border: 3px solid {FG} !important;
    border-radius: 0 !important;
    font-family: 'Comic Neue', cursive !important;
    font-weight: 900 !important;
    font-size: 0.95rem !important;
    text-transform: uppercase !important;
    letter-spacing: 1.5px !important;
    box-shadow: 5px 5px 0 {FG} !important;
    transition: all 0.06s ease !important;
}}
.stButton > button * {{ color: {FG} !important; }}
.stButton > button:hover {{
    background-color: {FG} !important;
    color: {BG} !important;
    box-shadow: 2px 2px 0 {FG} !important;
    transform: translate(3px, 3px) !important;
}}
.stButton > button:hover * {{ color: {BG} !important; }}
.stButton > button:active {{
    box-shadow: none !important;
    transform: translate(5px, 5px) !important;
}}
.stButton > button:disabled {{
    background-color: {BG2} !important;
    color: {MUTED} !important;
    border: 3px dashed {MUTED} !important;
    box-shadow: none !important;
}}
.stButton > button:disabled * {{ color: {MUTED} !important; }}

/* ── Alerts ── */
[data-testid="stAlert"] {{
    border: 3px solid {FG} !important;
    border-radius: 0 !important;
    box-shadow: 4px 4px 0 {FG} !important;
    background-color: {BG} !important;
}}
[data-testid="stAlert"] * {{
    color: {FG} !important;
    font-weight: 700 !important;
    background-color: transparent !important;
}}
[data-testid="stAlert"] svg,
[data-testid="stAlert"] svg path {{
    fill: {FG} !important;
    color: {FG} !important;
}}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {{
    border: 3px solid {FG} !important;
    border-radius: 0 !important;
    box-shadow: 6px 6px 0 {FG} !important;
    margin-bottom: 18px !important;
    background-color: {BG} !important;
    padding: 12px 16px !important;
}}
[data-testid="stChatMessage"] *,
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessage"] div,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] strong,
[data-testid="stChatMessage"] em {{
    color: {FG} !important;
    background-color: transparent !important;
}}

/* ── Bottom bar ── */
[data-testid="stBottom"],
[data-testid="stBottom"] > div,
[data-testid="stBottom"] > div > div {{
    background-color: {BG} !important;
    border-top: 4px solid {FG} !important;
}}

/* ── Chat input — kill ALL focus colors (the red outline) ── */
*:focus, *:focus-visible {{
    outline: none !important;
    box-shadow: none !important;
}}
[data-testid="stChatInput"],
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] > div > div {{
    background-color: {BG} !important;
    border: none !important;
    box-shadow: none !important;
}}
[data-testid="stChatInput"] textarea {{
    border: 3px solid {FG} !important;
    border-radius: 0 !important;
    background-color: {BG} !important;
    color: {FG} !important;
    font-family: 'Comic Neue', cursive !important;
    font-size: 1rem !important;
    box-shadow: 4px 4px 0 {FG} !important;
    caret-color: {FG} !important;
    outline: none !important;
}}
[data-testid="stChatInput"] textarea:focus,
[data-testid="stChatInput"] textarea:focus-visible {{
    border: 3px solid {FG} !important;
    box-shadow: 6px 6px 0 {FG} !important;
    outline: none !important;
}}
[data-testid="stChatInput"] textarea::placeholder {{
    color: {MUTED} !important;
    font-style: italic !important;
}}
/* Send button — hide ALL internal content (span icon-name text, SVG,
   anything Streamlit puts inside) and draw a pure-CSS arrow with ::after.
   This has zero dependency on Material Symbols font loading. */
[data-testid="stChatInput"] button {{
    background-color: {FG} !important;
    border: none !important;
    border-radius: 0 !important;
    width: 44px !important;
    min-width: 44px !important;
    height: 44px !important;
    overflow: hidden !important;
    padding: 0 !important;
    font-size: 0 !important;
    position: relative !important;
    flex-shrink: 0 !important;
}}
[data-testid="stChatInput"] button > * {{
    display: none !important;
}}
[data-testid="stChatInput"] button::after {{
    content: '' !important;
    display: block !important;
    position: absolute !important;
    top: 50% !important;
    left: 48% !important;
    width: 11px !important;
    height: 11px !important;
    border-top: 3px solid {BG} !important;
    border-right: 3px solid {BG} !important;
    transform: translate(-50%, -50%) rotate(45deg) !important;
}}

/* ── Progress bar ── */
[data-testid="stProgressBar"] {{
    border: 2px solid {FG} !important;
    border-radius: 0 !important;
    background-color: {BG} !important;
}}
[data-testid="stProgressBar"] > div {{
    background-color: {FG} !important;
    border-radius: 0 !important;
}}

/* ── Expander ── */
[data-testid="stExpander"] {{
    border: 3px solid {FG} !important;
    border-radius: 0 !important;
    box-shadow: 5px 5px 0 {FG} !important;
    overflow: hidden !important;
    background-color: {BG} !important;
}}
[data-testid="stExpander"] summary {{
    background-color: {FG} !important;
    padding: 8px 14px !important;
}}
/* Summary text label — Comic Neue on p/div only, NOT span.
   Streamlit puts the toggle arrow in a span using a Material Symbols
   ligature; overriding its font turns it into raw text. */
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary div {{
    background-color: transparent !important;
    color: {BG} !important;
    font-family: 'Comic Neue', cursive !important;
    font-weight: 900 !important;
    text-transform: uppercase !important;
    letter-spacing: 2px !important;
}}
/* Restore Material Symbols on every span inside summary so the
   expand/collapse arrow renders as an icon, not a word. */
[data-testid="stExpander"] summary span {{
    font-family: 'Material Symbols Rounded', 'Material Symbols Outlined',
                 'Material Icons', sans-serif !important;
    font-size: 20px !important;
    color: {BG} !important;
    -webkit-font-feature-settings: 'liga' !important;
    font-feature-settings: 'liga' !important;
    background-color: transparent !important;
    text-transform: none !important;
    letter-spacing: normal !important;
}}
[data-testid="stExpander"] summary svg,
[data-testid="stExpander"] summary svg * {{
    fill: {BG} !important;
    color: {BG} !important;
}}
[data-testid="stExpander"] > div {{
    background-color: {BG} !important;
}}
[data-testid="stExpander"] > div * {{
    color: {FG} !important;
}}

/* ── Spinner ── */
[data-testid="stSpinner"] > div {{
    border-color: {FG} transparent transparent !important;
}}

/* ── Divider ── */
hr {{
    border-color: {FG} !important;
    opacity: 0.4 !important;
}}

/* ── Scrollbar ── */
::-webkit-scrollbar {{ width: 8px; background: {BG2}; border-left: 2px solid {FG}; }}
::-webkit-scrollbar-thumb {{ background: {FG}; }}
</style>
""", unsafe_allow_html=True)

# ── Title ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="display:inline-block; background:{FG}; padding:10px 28px;
            box-shadow:7px 7px 0 {MUTED}; margin-bottom:4px;">
  <span class="inv">DOCUMENT Q&amp;A</span>
</div>
<div style="font-family:'Comic Neue',cursive; font-size:0.95rem; font-weight:700;
            font-style:italic; color:{MUTED}; margin-top:8px; margin-bottom:24px;">
  Upload docs &rarr; Ask questions &rarr; Get answers &nbsp;|&nbsp; Llama 3 + FAISS
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    toggle_label = "☀  Light Mode" if dark else "☾  Dark Mode"
    if st.button(toggle_label, key="dark_toggle"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

    st.divider()
    st.header("Upload Documents")

    uploaded_files = st.file_uploader(
        "PDF, DOCX, or TXT",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )

    overwrite = st.checkbox("Overwrite existing index", value=True)

    if st.button("Index Documents", disabled=not uploaded_files):
        if overwrite:
            clear_vectorstore()
            if os.path.exists(RAW_DIR):
                import shutil
                shutil.rmtree(RAW_DIR)
            st.session_state.messages = []
            st.session_state.pop("qa_chain", None)
            
        os.makedirs(RAW_DIR, exist_ok=True)
        progress = st.progress(0)
        for i, f in enumerate(uploaded_files):
            save_path = os.path.join(RAW_DIR, f.name)
            with open(save_path, "wb") as fh:
                fh.write(f.getbuffer())
            with st.spinner(f"Indexing {f.name}…"):
                chunks = ingest_file(save_path)
                add_to_vectorstore(chunks)
            progress.progress((i + 1) / len(uploaded_files))
        st.success(f"Indexed {len(uploaded_files)} file(s).")
        st.session_state.pop("qa_chain", None)
        st.rerun()

    if vectorstore_exists():
        st.info("Vector index is ready.")
        if st.button("Clear Index & Chat"):
            clear_vectorstore()
            if os.path.exists(RAW_DIR):
                import shutil
                shutil.rmtree(RAW_DIR)
            st.session_state.messages = []
            st.session_state.pop("qa_chain", None)
            st.rerun()
    else:
        st.warning("No index yet — upload and index a document first.")

# ── Main: Q&A ─────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Ask a question about your documents…"):
    if not vectorstore_exists():
        st.warning("Please upload and index at least one document first.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        if "qa_chain" not in st.session_state:
            with st.spinner("Loading index…"):
                vs = load_vectorstore()
                st.session_state.qa_chain = build_qa_chain(vs)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Thinking…"):
                    result = st.session_state.qa_chain.invoke({"query": question})
                answer = result["result"]
                sources = result.get("source_documents", [])
            except Exception as e:
                answer = f"**Error getting answer:** {e}"
                sources = []
            st.markdown(answer)

            if sources:
                with st.expander("Sources"):
                    for doc in sources:
                        meta = doc.metadata
                        label = meta.get("source", "unknown")
                        page = meta.get("page")
                        ref = f"**{os.path.basename(label)}**"
                        if page is not None:
                            ref += f" — page {page + 1}"
                        st.markdown(ref)
                        st.caption(doc.page_content[:300] + "…")

        st.session_state.messages.append({"role": "assistant", "content": answer})
