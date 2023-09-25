from st_on_hover_tabs import on_hover_tabs
import streamlit as st

st.set_page_config(layout="wide")

st.header("Custom tab component for on-hover navigation bar")
st.markdown('<style>' + open('./style.css').read() + '</style>', unsafe_allow_html=True)

with st.sidebar:
    tabs = on_hover_tabs(tabName=['Home', 'EDA','Topic Analysis','Subtopics', 'Likes','Dislikes','Sentiment','Competition'], 
                         iconName=['🏠', '📊','💡', '📝','👍','👎','💭','🏆'], 
                         styles = {'navtab': {'background-color':'#111',
                                                  'color': '#818181',
                                                  'font-size': '16px',
                                                  'transition': '.3s',
                                                  'white-space': 'nowrap',
                                                  'text-transform': 'uppercase'},
                                   'tabOptionsStyle': {':hover :hover': {'color': 'white',
                                                                      'cursor': 'pointer'}},
                                   'iconStyle':{'position':'fixed',
                                                    'left':'7.5px',
                                                    'text-align': 'left'},
                                   'tabStyle' : {'list-style-type': 'none',
                                                     'margin-bottom': '30px',
                                                     'padding-left': '30px'}},
                         key="1")

# if tabs =='Home':
#     st.title("Home")
#     st.write('Name of option is {}'.format(tabs))

# elif tabs == 'Money':
#     st.title("Paper")
#     st.write('Name of option is {}'.format(tabs))

# elif tabs == 'Economy':
#     st.title("Tom")
#     st.write('Name of option is {}'.format(tabs))
    
