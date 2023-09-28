import streamlit as st
import altair as alt
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
# product = st.selectbox("", products)
product = sac.chip(
    items= products, index=[0, 2], format_func='title', align='left', return_index=True, multiple=False
)
if product == 0 : 
   product = 'Suzuki'
elif product ==1 : 
     product = 'Honda'
elif product ==2 : 
     product = 'TVS'
# st.write(product)
#read data
df = pd.read_csv('All_Reviews.csv')
df.columns = ['Review', 'Brand']

df['Clean_Comment'] = df['Review'].apply(clean_text)
topics = pd.read_csv('topic_count1.csv')
topics["count"] = topics["count"]
topics["Topics"] = topics["Subtopic"]
# Filter data based on selected product
filtered_df = df[df['Brand'] == product]
filtered_raw_df = df[df['Brand'] == product].head(1000)

# Get all comments for selected product
text = ' '.join(df[df['Brand'] == product]['Clean_Comment'])

st.markdown('<style>' + open('./style.css').read() + '</style>', unsafe_allow_html=True)
# sac.divider(label='🏠', align='center')

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
        st.write(f'<h1 style="margin-top:-40px;text-align: left;color:#094780;font-size:30px;">{"Machine Learning Techniques Used:"}</h1>', unsafe_allow_html=True)
        st.write(f'<h1 style="text-align: left;color:#EC2A39;font-size:20px;">{"Topic Modeling"}</h1>', unsafe_allow_html=True)
        st.info(
        """
        Topic modeling is a machine learning technique used to analyze and categorize large volumes of text data. It identifies recurring patterns or themes, known as 'topics,' within the text.

        For example, if a business receives numerous customer reviews, the topic modeling algorithm would identify words that commonly co-occur and form topics based on these patterns. This could reveal topics like product quality, customer service, or pricing.
        """
        )
        st.write(f'<h1 style="text-align: left;color:#EC2A39;font-size:20px;">{"NLP - Natural Language Processing"}</h1>', unsafe_allow_html=True)
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
        st.write(f'<h1 style="margin-top:-20px;text-align: left;color:#9FACB8;font-size:25px;">{"Data Collected For Product Analysis:"}</h1>', unsafe_allow_html=True)
        st.write(f'<h1 style="margin-top:-20px;text-align: left;color:#9FACB8;font-size:20px;">{"Word Cloud"}</h1>', unsafe_allow_html=True)
        if product == 'Suzuki':
            st.image('suzuki_wordcloud.png', use_column_width=True)
        if product == 'Honda':
            st.image('Honda_wordcloud.png', use_column_width=True)
        if product == 'TVS':
            st.image('tvs_wordcloud.png', use_column_width=True)
        # st.write('Add wcloud and some charts'.format(tabs))
        # Display raw data
        st.write(f'<h1 style="margin-top:-20px;text-align: left;color:#9FACB8;font-size:25px;">{"Raw Data"}</h1>', unsafe_allow_html=True)
        
        st.dataframe(filtered_raw_df.head(100))
    

elif tabs == 'Topic Analyzer':
        sac.divider(label='💡', align='center')
        st.write(f'<h1 style="margin-top:-20px;color:#094780;font-size:30px;">{"Topic Analyzer"}</h1>', unsafe_allow_html=True)
        # st.title("Topics")
        st.write(f'<h1 style="margin-top:-20px;text-align: left;color:#9FACB8;font-size:25px;">{"Extracted SubTopics and Review Count"}</h1>', unsafe_allow_html=True)
        if product == 'Suzuki':
            col1 , col2 = st.columns(2)
            product_b = pd.read_csv('Suzuki_subtopic_topic.csv')
            product_b = product_b[['Subtopic','Count']]
            product_b['Count']=product_b['Count'].astype('int')
            
            # chart_data = pd.DataFrame(product_b['Count'],index=product_b['Subtopic'])
            # chart_data = pd.DataFrame(product_b, columns=["Subtopic"])
            # st.bar_chart(chart_data)
            # Vertical stacked bar chart
            col1.dataframe(product_b)
            col2.bar_chart(product_b, x= "Subtopic",y="Count")

        if product == 'Honda':
            col1 , col2 = st.columns(2)
            product_a = pd.read_csv('Honda_subtopic_topic.csv')
            product_a = product_a[['Subtopic','Count']]
            col1.dataframe(product_a)
            col2.bar_chart(product_a, x= "Subtopic",y="Count")

        if product == 'TVS':
            col1 , col2 = st.columns(2)
            product_a = pd.read_csv('TVS_subtopic_topic.csv')
            product_a = product_a[['Subtopic','Count']]
            col1.dataframe(product_a)
            col2.bar_chart(product_a, x= "Subtopic",y="Count")
            
        st.write(f'<h1 style="margin-top:-20px;text-align: left;color:#9FACB8;font-size:25px;">{"7 key Topic Considered"}</h1>', unsafe_allow_html=True)
        col1 , col2 , col3 , col4 , col5= st.columns(5)
        col1.info('👀 Body / Design / Looks/ Style')
        col2.info('⚡ Engine / Performance / Speed')
        col3.info('🛠️ Service & Maintenance')
        col4.info('🆕 Special Feature, New feature')
        col5.info('🥇 Competition')
        col3.info('🏍️ Ride experience / Comfortability')
        col4.info('💲 Price, Cost, Buying')
    
        st.write(f'<h1 style="margin-top:-20px;text-align: left;color:#9FACB8;font-size:25px;">{"Topic Insights For Brand: Likes & Dislikes"}</h1>', unsafe_allow_html=True)
        # if product == 'Suzuki':
        col1, col2 = st.columns(2)
        # with st.expander("Click to see insights"):
        if product == 'Suzuki':
                col1.write(f'<h1 style="margin-top:-20px;text-align: center;color:#9FACB8;font-size:20px;">{"Likes👍"}</h1>', unsafe_allow_html=True)
                col1.info('''
                - **Riding Experience and Road Conditions**: Riders appreciate the smooth handling and comfortable suspension on various road surfaces.
                - Vehicle Buying Experience: Positive feedback is received for dealerships with straightforward purchasing processes and friendly sales staff.
                - Vehicle Maintenance and Component Considerations: Owners like the durability of components and reasonable maintenance costs.
                - Positive Vehicle Experience and Appreciation: Customers express loyalty to the brand and share memorable riding experiences.
                - Comfort and Quality of Scooter Seating, particularly for long rides: Owners enjoy comfortable seating and ergonomic design for extended journeys.''')

                # Customer preferences
                col2.write(f'<h1 style="margin-top:-20px;text-align: center;color:#9FACB8;font-size:20px;">{"Dislikes👎"}</h1>', unsafe_allow_html=True)
                col2.info('''
                - Vehicle Starting Issues and Engine Problems: Some users report occasional starting problems and engine issues, which can be frustrating.
                - Vehicle Body and Design Considerations: There are dislikes related to limited color options and outdated design in certain models.
                - Mileage and Scooter Comparison, with emphasis on Honda Activa: Some customers are disappointed by lower-than-expected mileage and unfavorable comparisons with Honda Activa.
                - Issues and Experiences with Scooter Service and Performance: Negative experiences are reported, including frequent service visits and unresolved problems.
                - Features and Buying Considerations: Some buyers express disappointment with limited feature choices and uninformed purchases.''')
        if product == 'Honda':
                col1.write(f'<h1 style="margin-top:-20px;text-align: center;color:#9FACB8;font-size:20px;">{"Likes👍"}</h1>', unsafe_allow_html=True)
                col1.info('''
                - Comparing the Honda Activa's Engine and Riding Experience: Enthusiasts appreciate the comparison of engine performance and riding experience, seeking information to make informed choices.
                - Exploring the Best Features and Colors for an Awesome and Comfortable Ride: Riders value information on features and colors that enhance the comfort and enjoyment of their motorcycle experience.
                - Maximizing Mileage and Performance: Riders seek tips on improving engine efficiency, lighting, and smooth riding to enhance their overall motorcycle experience.
                - Awesome Look and Affordable Price Range of Honda Motorcycles: Buyers appreciate the combination of attractive design and affordability in Honda motorcycles.
                - Choosing the Perfect Scooty: Prospective buyers are interested in factors to consider when making a scooter purchase, indicating a desire for informed decision-making. ''')

                # Customer preferences
                col2.write(f'<h1 style="margin-top:-20px;text-align: center;color:#9FACB8;font-size:20px;">{"Dislikes👎"}</h1>', unsafe_allow_html=True)
                col2.info('''
                - Honda Scooter Ownership: Some riders may face challenges and common problems with Honda scooters, suggesting potential negative experiences.
                - Powerful and Attractive Commuter Bikes: There may be concerns or dislikes related to commuter bikes in terms of performance or styling.
                - Troubleshooting Common Motorcycle Problems: Riders encounter issues related to mileage, speed, and engine performance, which can be seen as negative experiences.
                - Optimizing Your Honda Motorcycle Service Experience: Riders may have concerns about managing service costs and maintenance, potentially reflecting negative aspects of the ownership experience.
                - Honda Motorcycle Gear Shift Issues: Some riders may experience gear shift problems, impacting the smoothness of their rides.''')
        if product == 'Tvs':
                col1.write(f'<h1 style="margin-top:-20px;text-align: center;color:#9FACB8;font-size:20px;">{"Likes👍"}</h1>', unsafe_allow_html=True)
                col1.info('''
                - Maximizing the Value of Your Motorcycle: Riders appreciate motorcycles that offer superb power, great rides, and smart money choices.
                - Indian Motorcycle Machines: Enthusiasts enjoy the journey of feel, the competitive pricing, and the riding pleasure that Indian motorcycles provide.
                - Striking the Perfect Balance in Motorcycle Ownership: Riders value motorcycles that offer good rides, ideal prices, and strong performance, indicating a desire for well-rounded options.
                - Unleashing Awesome Rides and Sporty Looks: Motorcycle enthusiasts favor motorcycles that offer exciting rides and sporty aesthetics.
                - The Scooter Revolution: Riders appreciate fuel efficiency, stylish rides, and positive rider experiences when it comes to scooters.''')

                # Customer preferences
                col2.write(f'<h1 style="margin-top:-20px;text-align: center;color:#9FACB8;font-size:20px;">{"Dislikes👎"}</h1>', unsafe_allow_html=True)
                col2.info('''
               - Navigating Service Challenges and Making Informed Buying Decisions: Some riders express frustration with service challenges, and there is a need for better information to make informed buying decisions.
                - Troubleshooting Vehicle Issues, Starting Strong, and Self-Improvement: Riders face challenges with troubleshooting vehicle issues and starting problems, which can be seen as negative experiences.
                - The Art of Riding: There are concerns or dislikes related to aspects of motorcycle aesthetics or style.
                - Elevating the Scooter Experience: Some riders may have reservations about the scooter experience, possibly due to expectations of greater innovation.
                - Navigating the World of Motorcycle Ownership: Negative experiences are reported, including challenges with service, maintenance, and the overall rider's journey.''')

elif tabs == 'Sentiment Analysis':
        sac.divider(label='💭', align='center')
        st.title("Customer Sentiment")
        st.write('Customer Sentiment'.format(tabs))

elif tabs == 'Competitive Analysis':
        sac.divider(label='🏆', align='center')
        st.title("Competition")
        st.write('Competition - Suzuki/Honda/TVS'.format(tabs))
    
