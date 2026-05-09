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

        .stApp {
            background: var(--page-bg);
            color: var(--text);
        }

        [data-testid="stHeader"] {
            background: rgba(246, 247, 248, 0.96);
            border-bottom: 1px solid var(--border);
        }

        [data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid var(--border);
        }

        [data-testid="stSidebar"] * {
            color: var(--text);
        }

        h1, h2, h3, h4, h5, h6, p, label, span, div {
            letter-spacing: 0;
        }

        h1, h2, h3 {
            color: #111827;
        }

        .stMarkdown, .stText, .stCaption, label, p {
            color: var(--text);
        }

        small, [data-testid="stCaptionContainer"] {
            color: var(--muted);
        }

        [data-testid="stFileUploader"],
        [data-testid="stExpander"],
        [data-testid="stForm"],
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--panel-bg);
            border-color: var(--border);
            border-radius: 8px;
        }

        .stAlert {
            background: #ffffff;
            border: 1px solid var(--border);
            color: var(--text);
        }

        .stButton > button,
        .stDownloadButton > button,
        button[kind="primary"] {
            background: var(--accent);
            border: 1px solid var(--accent-strong);
            color: #ffffff;
            border-radius: 6px;
            font-weight: 600;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            background: var(--accent-strong);
            border-color: var(--accent-strong);
            color: #ffffff;
        }

        input, textarea, [data-baseweb="select"] > div {
            background: #ffffff;
            color: var(--text);
            border-color: var(--border);
        }

        hr {
            border-color: var(--border);
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
