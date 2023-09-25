from st_on_hover_tabs import on_hover_tabs
import streamlit as st
import streamlit_antd_components as sac
st.set_page_config(layout="wide")
st.write(f'<h1 style="margin-top:-90px;color:#094780;font-size:35px;">{"VroomViews🏍️"}</h1>', unsafe_allow_html=True)
# st.write("Turn every review into a pit stop for improvement with our Automotive Review Analysis App – where user feedback fuels your success.")
st.write(f'<h1 style="margin-top:-55px;color:#EC2A39;font-size:15px;">{"Turn every review into a pit stop for improvement with our Automotive Review Analysis App – where user feedback fuels your success."}</h1>', unsafe_allow_html=True)
# st.markdown("""<hr style="height:1px;border:none;color:#9FACB8;background-color:#9FACB8;" /> """, unsafe_allow_html=True)
# sac.divider(label='', align='center')
st.write(f'<h1 style="margin-top:-20px;text-align: left;color:#9FACB8;font-size:15px;">{"Toggle between the selected brands to derive insights for different Key takeaways:"}</h1>', unsafe_allow_html=True)
sac.chip(
    items=[
        sac.ChipItem(label='Suzuki', icon='motorcycle'),
        sac.ChipItem(label='🏍 Honda', icon='racing'),
        sac.ChipItem(label='TVS', icon='bike'),
    ], index=[0, 2], format_func='title', align='left', return_index=True
)
st.markdown('<style>' + open('./style.css').read() + '</style>', unsafe_allow_html=True)
# sac.divider(label='🏠', align='center')

with st.sidebar:
    tabs = on_hover_tabs(tabName=['Home', 'EDA','Topic Analysis','Subtopics', 'Likes','Dislikes','Sentiment','Competition'], 
                         iconName=['🏠', '📊','💡', '📝','👍','👎','💭','🏆'], 
                         styles = {'navtab': {'background-color':'#083d6e',
                                                  'color': '#adb0b3',
                                                  'font-size': '17px',
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
        st.write(f'<h1 style="margin-top:-300px;color:#094780;font-size:35px;">{""}</h1>', unsafe_allow_html=True)
        sac.divider(label='🏠', align='center')
        st.write(f'<h1 style="margin-top:-40px;text-align: center;color:#094780;font-size:15px;">{"Key Take Aways : Topics being discussed | Likes in the SKUs | Dislikes in the SKUs | Customer Sentiment | Competitive Analysis | Major Keywords | Subtopics across different automotive Key Factors"}</h1>', unsafe_allow_html=True)
        # st.write(f'<h1 style="margin-top:-20px;text-align: center;color:#9FACB8;font-size:15px;">{"Toggle between the selected brands to derive insights for different Key takeaways:"}</h1>', unsafe_allow_html=True)
        st.image('Picture1.png', use_column_width=True)
        col11, col22, col33 = st.columns(3)
        
    # st.write('Suzuki/Honda/TVS'.format(tabs))

elif tabs == 'EDA':
        sac.divider(label='📊', align='center')
        st.write(f'<h1 style="margin-top:-20px;color:#094780;font-size:30px;">{"Exploratory Data Analysis (EDA)"}</h1>', unsafe_allow_html=True)
        # st.write(f'<h1 style="margin-top:-20px;color:#9FACB8;font-size:15px;">{"Toggle between the selected brands to derive insights for different Key takeaways:"}</h1>', unsafe_allow_html=True)
        # tab1, tab2, tab3 = st.tabs( ["Suzuki", "Honda", "TVS"])
        st.write('Add wcloud and some charts'.format(tabs))
    

elif tabs == 'Topic Analysis':
        sac.divider(label='💡', align='center')
        st.title("Topics")
        st.write('7 key topics'.format(tabs))

elif tabs == 'Subtopics':
        sac.divider(label='📝', align='center')
        st.title("Subtopics")
        st.write('20 sub topics'.format(tabs)) 

elif tabs == 'Likes':
        sac.divider(label='👍', align='center')
        st.title("Likes")
        st.write('Likes/topic'.format(tabs))

elif tabs == 'Dislikes':
        sac.divider(label='👎', align='center')
        st.title("Dislikes")
        st.write('Dislikes/topic'.format(tabs))

elif tabs == 'Sentiment':
        sac.divider(label='💭', align='center')
        st.title("Customer Sentiment")
        st.write('Customer Sentiment'.format(tabs))

elif tabs == 'Competition':
        sac.divider(label='🏆', align='center')
        st.title("Competition")
        st.write('Competition - Suzuki/Honda/TVS'.format(tabs))
    
