from st_on_hover_tabs import on_hover_tabs
import streamlit as st
import streamlit_antd_components as sac
st.set_page_config(layout="wide")
st.header("Automotive Customer Review Analysis")
# st.write("Turn every review into a pit stop for improvement with our Automotive Review Analysis App – where user feedback fuels your success.")
st.write(f'<h1 style="margi-top:-20px,color:#EC2A39;font-size:15px;">{"Turn every review into a pit stop for improvement with our Automotive Review Analysis App – where user feedback fuels your success."}</h1>', unsafe_allow_html=True)
# st.markdown("""<hr style="height:1px;border:none;color:#9FACB8;background-color:#9FACB8;" /> """, unsafe_allow_html=True)
st.markdown('<style>' + open('./style.css').read() + '</style>', unsafe_allow_html=True)
sac.divider(label='🏠', align='center')

with st.sidebar:
    tabs = on_hover_tabs(tabName=['Home', 'EDA','Topic Analysis','Subtopics', 'Likes','Dislikes','Sentiment','Competition'], 
                         iconName=['🏠', '📊','💡', '📝','👍','👎','💭','🏆'], 
                         styles = {'navtab': {'background-color':'#111',
                                                  'color': '#08',
                                                  'font-size': '16px',
                                                  'transition': '.3s',
                                                  'white-space': 'nowrap',
                                                  'text-transform': 'uppercase'},
                                   'tabOptionsStyle': {':hover :hover': {'color': 'white',
                                                                      'cursor': 'pointer'}},
                                   'iconStyle':{'position':'fixed',
                                                    'left':'6.5px',
                                                    'text-align': 'left'},
                                   'tabStyle' : {'list-style-type': 'none',
                                                     'margin-bottom': '30px',
                                                     'padding-left': '30px'}},
                         default_choice=0)


if tabs =='Home':
        st.write(f'<h1 style="text-align: center;color:#094780;font-size:15px;">{"Key Take Aways : Topics being discussed | Likes in the SKUs | Dislikes in the SKUs | Customer Sentiment | Competitive Analysis | Major Keywords | Subtopics across different automotive Key Factors"}</h1>', unsafe_allow_html=True)
        st.write(f'<h1 style="text-align: center;color:#9FACB8;font-size:15px;">{"Select the Brand for which you would like to see the Report :"}</h1>', unsafe_allow_html=True)
        st.image('Picture1.png', use_column_width=True)
        col11, col22, col33 = st.columns(3)
        
    # st.write('Suzuki/Honda/TVS'.format(tabs))

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
    
