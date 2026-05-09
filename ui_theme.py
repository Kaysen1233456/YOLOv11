import streamlit as st


def apply_light_print_theme() -> None:
    """Apply a light, print-friendly Streamlit theme."""
    st.markdown(
        """
        <style>
        :root {
            --page-bg: #f6f7f8;
            --panel-bg: #ffffff;
            --border: #d8dde3;
            --text: #1f2933;
            --muted: #52606d;
            --accent: #1f4e79;
            --accent-strong: #143a5a;
        }

        html, body,
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        .main {
            background: var(--page-bg) !important;
            color: var(--text) !important;
        }

        .block-container {
            background: transparent !important;
            color: var(--text) !important;
        }

        [data-testid="stHeader"] {
            background: rgba(246, 247, 248, 0.96) !important;
            border-bottom: 1px solid var(--border) !important;
        }

        [data-testid="stSidebar"] {
            background: #ffffff !important;
            border-right: 1px solid var(--border) !important;
        }

        [data-testid="stSidebar"] * {
            color: var(--text) !important;
        }

        h1, h2, h3, h4, h5, h6, p, label, span, div {
            letter-spacing: 0;
        }

        h1, h2, h3 {
            color: #111827 !important;
        }

        .stMarkdown, .stText, .stCaption, label, p {
            color: var(--text) !important;
        }

        small, [data-testid="stCaptionContainer"] {
            color: var(--muted) !important;
        }

        [data-testid="stFileUploader"],
        [data-testid="stExpander"],
        [data-testid="stForm"],
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--panel-bg) !important;
            border-color: var(--border) !important;
            border-radius: 8px !important;
        }

        .stAlert {
            background: #ffffff !important;
            border: 1px solid var(--border) !important;
            color: var(--text) !important;
        }

        .stButton > button,
        .stDownloadButton > button,
        button[kind="primary"] {
            background: var(--accent) !important;
            border: 1px solid var(--accent-strong) !important;
            color: #ffffff !important;
            border-radius: 6px !important;
            font-weight: 600 !important;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            background: var(--accent-strong) !important;
            border-color: var(--accent-strong) !important;
            color: #ffffff !important;
        }

        input, textarea,
        [data-baseweb="input"] input,
        [data-baseweb="textarea"],
        [data-baseweb="select"] > div {
            background: #ffffff !important;
            color: var(--text) !important;
            border-color: var(--border) !important;
        }

        [data-baseweb="radio"] *,
        [data-testid="stWidgetLabel"] *,
        [data-testid="stMarkdownContainer"] * {
            color: var(--text) !important;
        }

        hr {
            border-color: var(--border) !important;
        }

        @media print {
            [data-testid="stSidebar"],
            [data-testid="stHeader"],
            [data-testid="stToolbar"],
            .stButton,
            .stFileUploader {
                display: none !important;
            }

            .stApp,
            .main,
            section,
            div {
                background: #ffffff !important;
                color: #000000 !important;
                box-shadow: none !important;
                text-shadow: none !important;
            }

            h1, h2, h3, h4, h5, h6, p, span, label {
                color: #000000 !important;
            }

            .stAlert,
            [data-testid="stExpander"],
            [data-testid="stVerticalBlockBorderWrapper"] {
                border: 1px solid #777777 !important;
                background: #ffffff !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
