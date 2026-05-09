import streamlit as st

from final_system import login_page, main_app
from ui_theme import apply_light_print_theme


apply_light_print_theme()


if __name__ == "__main__":
    if st.session_state.logged_in:
        main_app()
    else:
        login_page()
