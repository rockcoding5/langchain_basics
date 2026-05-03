# Streamlit web app for restaurant name and menu generation
import streamlit as st
from utils import langchain_helper as lh

# App title
st.title("Restaurant Name Generator")

# Sidebar for cuisine selection
cuisine = st.sidebar.selectbox(
    "Pick a Cuisine",
    ("Indian", "Mexican", "Italian", "Arabian", "American")
    )

# Generate content when cuisine is selected
if cuisine:
    # Get restaurant name and menu items from helper
    response = lh.get_rest_name_item(cuisine=cuisine)

    # Display restaurant name as header
    st.header(response['restaurant_name'])

    # Process and display menu items
    menu_items = [item.strip() for item in response['menu_items'].split(",")]

    st.write("** Menu Items **")

    # List each menu item
    for item in menu_items:
        st.write("-> ", item)
