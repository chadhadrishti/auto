from st_on_hover_tabs import on_hover_tabs
import streamlit as st

st.set_page_config(layout="wide")

st.header(f'<h1 style="color:#094780;font-size:22px;">{"Automotive Customer Review Analysis"}</h1>', unsafe_allow_html=True)
st.write("Turn every review into a pit stop for improvement with our Automotive Review Analysis App – where user feedback fuels your success.")
st.markdown(f'<h1 style="color:#EC2A39;font-size:15px;">{"Turn every review into a pit stop for improvement with our Automotive Review Analysis App – where user feedback fuels your success."”}</h1>', unsafe_allow_html=True)
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
                         default_choice=0)


if tabs =='Home':
        st.title("brands")
        st.write('Suzuki/Honda/TVS'.format(tabs))

elif tabs == 'EDA':
        st.title("EDA")
        st.write('Add wcloud and some charts'.format(tabs))

elif tabs == 'Topic Analysis':
        st.title("Topics")
        st.write('7 key topics'.format(tabs))

elif tabs == 'Subtopics':
        st.title("Subtopics")
        st.write('20 sub topics'.format(tabs)) 

elif tabs == 'Likes':
        st.title("Likes")
        st.write('Likes/topic'.format(tabs))

elif tabs == 'Dislikes':
        st.title("Dislikes")
        st.write('Dislikes/topic'.format(tabs))

elif tabs == 'Sentiment':
        st.title("Customer Sentiment")
        st.write('Customer Sentiment'.format(tabs))

elif tabs == 'Competition':
        st.title("Competition")
        st.write('Competition - Suzuki/Honda/TVS'.format(tabs))
    
