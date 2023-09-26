import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import nltk
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.agents import load_tools, Tool
from langchain.agents import ZeroShotAgent, AgentExecutor
from langchain import OpenAI, LLMChain
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
from streamlit import components
from pyecharts import options as opts
from pyecharts.charts import Pie
from pyecharts import options as opts
from pyecharts.charts import Polar
from pyecharts.globals import ThemeType
import plotly.graph_objects as go
from PIL import Image

from streamlit import config
import re
import os

from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import UnstructuredExcelLoader
from langchain.vectorstores import FAISS
import tempfile

st.set_option("deprecation.showfileUploaderEncoding", False)
import pandas as pd
from st_on_hover_tabs import on_hover_tabs
import streamlit as st
import streamlit_antd_components as sac
st.set_page_config(layout="wide")

# Function to clean the text
def clean_text(text):
    # Check if the input is a string
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenize text
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Perform lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Join the tokens back into a single string
    text = ' '.join(tokens)

    return text
    
st.write(f'<h1 style="margin-top:-90px;color:#094780;font-size:35px;">{"VroomViews🏍️"}</h1>', unsafe_allow_html=True)
# st.write("Turn every review into a pit stop for improvement with our Automotive Review Analysis App – where user feedback fuels your success.")
st.write(f'<h1 style="margin-top:-55px;color:#EC2A39;font-size:15px;">{"Turn every review into a pit stop for improvement with our Automotive Review Analysis App – where user feedback fuels your success."}</h1>', unsafe_allow_html=True)
# st.markdown("""<hr style="height:1px;border:none;color:#9FACB8;background-color:#9FACB8;" /> """, unsafe_allow_html=True)
# sac.divider(label='', align='center')
st.write(f'<h1 style="margin-top:-20px;text-align: left;color:#9FACB8;font-size:15px;">{"Toggle between the selected brands to derive insights for different Key takeaways:"}</h1>', unsafe_allow_html=True)
products = ['Suzuki','Honda','TVS']
product = sac.chip(
    items=[
        sac.ChipItem(label='Suzuki', icon='bike'),
        sac.ChipItem(label='Honda', icon='racing'),
        sac.ChipItem(label='TVS', icon='bike'),
    ], index=[0, 2], format_func='title', align='left', return_index=True
)
# Filter data based on selected product
filtered_df = df[df['Brand'] == product]
filtered_raw_df = df[df['Brand'] == product].head(1000)

# Get all comments for selected product
text = ' '.join(df[df['Brand'] == product]['Clean_Comment'])

st.markdown('<style>' + open('./style.css').read() + '</style>', unsafe_allow_html=True)
# sac.divider(label='🏠', align='center')
#read data
df = pd.read_csv('All_Reviews.csv')
df.columns = ['Review', 'Brand']

df['Clean_Comment'] = df['Review'].apply(clean_text)
topics = pd.read_csv('topic_count1.csv')
topics["count"] = topics["count"]
topics["Topics"] = topics["Subtopic"]

with st.sidebar:
    tabs = on_hover_tabs(tabName=['Home','Approach', 'EDA','Topic Analyzer','Sentiment Analysis','Competitive Analysis'], 
                         iconName=['🏠','🚀', '📊','💡','💭','🏆'], 
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
                                                     'margin-bottom': '40px',
                                                     'padding-left': '30px'}},
                         default_choice=0)


if tabs =='Home':
        st.write(f'<h1 style="margin-top:-300px;color:#094780;font-size:35px;">{""}</h1>', unsafe_allow_html=True)
        sac.divider(label='🏠', align='center')
        st.write(f'<h1 style="margin-top:-40px;text-align: center;color:#094780;font-size:15px;">{"Key Take Aways : Topics being discussed | Likes in the SKUs | Dislikes in the SKUs | Customer Sentiment | Competitive Analysis | Major Keywords | Subtopics across different automotive Key Factors"}</h1>', unsafe_allow_html=True)
        # st.write(f'<h1 style="margin-top:-20px;text-align: center;color:#9FACB8;font-size:15px;">{"Toggle between the selected brands to derive insights for different Key takeaways:"}</h1>', unsafe_allow_html=True)
        st.image('Picture1.png', use_column_width=True)
        # col11, col22, col33 = st.columns(3)

elif tabs =='Approach':
        st.write(f'<h1 style="margin-top:-300px;color:#094780;font-size:35px;">{""}</h1>', unsafe_allow_html=True)
        sac.divider(label='🚀', align='center')
        # st.write(f'<h1 style="margin-top:-40px;text-align: center;color:#094780;font-size:15px;">{"Key Take Aways : Topics being discussed | Likes in the SKUs | Dislikes in the SKUs | Customer Sentiment | Competitive Analysis | Major Keywords | Subtopics across different automotive Key Factors"}</h1>', unsafe_allow_html=True)
        # st.write(f'<h1 style="margin-top:-20px;text-align: center;color:#9FACB8;font-size:15px;">{"Toggle between the selected brands to derive insights for different Key takeaways:"}</h1>', unsafe_allow_html=True)
        st.image('Approach2.png', use_column_width=True)
        st.header('Machine Learning Techniques Used:')
        st.subheader("Topic Modeling")
        st.info(
        """
        Topic modeling is a machine learning technique used to analyze and categorize large volumes of text data. It identifies recurring patterns or themes, known as 'topics,' within the text.

        For example, if a business receives numerous customer reviews, the topic modeling algorithm would identify words that commonly co-occur and form topics based on these patterns. This could reveal topics like product quality, customer service, or pricing.
        """
        )
        st.subheader("NLP - Natural Language Processing")
        st.info(
        """
        Natural Language Processing (NLP) is a subfield of artificial intelligence (AI) that focuses on the interaction between computers and human language. Its primary goal is to enable computers to understand, interpret, and generate human language in a valuable way. NLP combines techniques from computer science, linguistics, and machine learning to process and analyze text and speech data.
        """
        )
    # st.write('Suzuki/Honda/TVS'.format(tabs))

elif tabs == 'EDA':
        sac.divider(label='📊', align='center')
        st.write(f'<h1 style="margin-top:-20px;color:#094780;font-size:30px;">{"Exploratory Data Analysis (EDA)"}</h1>', unsafe_allow_html=True)
        # st.write(f'<h1 style="margin-top:-20px;color:#9FACB8;font-size:15px;">{"Toggle between the selected brands to derive insights for different Key takeaways:"}</h1>', unsafe_allow_html=True)
        # tab1, tab2, tab3 = st.tabs( ["Suzuki", "Honda", "TVS"])
        st.subheader("Data Collected For Product Analysis:")
        st.subheader("Word Cloud")
        if items == 'Suzuki':
            st.image('suzuki_wordcloud.png', width=500)
        if items == 'Honda':
            st.image('Honda_wordcloud.png', width=500)
        if items == 'TVS':
            st.image('tvs_wordcloud.png', width=500)
        # st.write('Add wcloud and some charts'.format(tabs))
        # Display raw data
        st.subheader("Raw Data")
        
        st.dataframe(filtered_raw_df.head(100))
    

elif tabs == 'Topic Analyzer':
        sac.divider(label='💡', align='center')
        st.title("Topics")
        st.write('7 key topics'.format(tabs))

elif tabs == 'Sentiment Analysis':
        sac.divider(label='💭', align='center')
        st.title("Customer Sentiment")
        st.write('Customer Sentiment'.format(tabs))

elif tabs == 'Competitive Analysis':
        sac.divider(label='🏆', align='center')
        st.title("Competition")
        st.write('Competition - Suzuki/Honda/TVS'.format(tabs))
    
