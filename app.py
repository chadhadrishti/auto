from st_on_hover_tabs import on_hover_tabs
import streamlit as st

st.set_page_config(layout="wide")

st.header("Custom tab component for on-hover navigation bar")
# st.markdown('<style>' + open('./style.css').read() + '</style>', unsafe_allow_html=True)


with st.sidebar:
    tabs = on_hover_tabs(tabName=['Home', 'EDA','Topic Analysis','Subtopics', 'Likes','Dislikes','Sentiment','Competition'], 
                         iconName=['🏠', '📊','💡', '📝','👍','👎','💭','🏆'], default_choice =1)

# if tabs =='Dashboard':
#     st.title("Navigation Bar")
#     st.write('Name of option is {}'.format(tabs))

# elif tabs == 'Money':
#     st.title("Paper")
#     st.write('Name of option is {}'.format(tabs))

# elif tabs == 'Economy':
#     st.title("Tom")
#     st.write('Name of option is {}'.format(tabs))
    
