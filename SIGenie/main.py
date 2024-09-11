import streamlit as st
import os

# Set page config (this should be the first Streamlit command)
st.set_page_config(layout="wide", page_title="Shipping Document Viewer")

# Apply custom font
custom_font_css = """
<style>
@font-face {
    font-family: 'Freesentation';
    src: url('./fonts/Freesentation.ttf') format('truetype');
}

* {
    font-family: 'Freesentation', sans-serif;
}
</style>
"""
st.markdown(custom_font_css, unsafe_allow_html=True)

# Import modules after setting page config
import json_bkg
import json_si
import json_bl  # Changed from JSON_BL to json_bl

def main():
    # Sidebar menu
    menu = st.sidebar.selectbox(
        "Select Document Type",
        ["Booking", "Shipping Instructions", "Bill of Lading"]
    )

    # Main content
    if menu == "Booking":
        st.title("Booking Viewer")
        json_bkg.main()
    elif menu == "Shipping Instructions":
        st.title("Shipping Instructions Viewer")
        json_si.main()
    elif menu == "Bill of Lading":
        st.title("Bill of Lading Viewer")
        json_bl.main()  # Changed from JSON_BL to json_bl

    # Footer 추가
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray; padding: 10px;'>"
        "Copyright © 2024 SIGenie 0.01 - Early Access Version. All rights reserved."
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()