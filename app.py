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
        st.write(f'<h1 style="margin-top:-15px;color:#094780;font-size:15px;">{"Above Subtopics were further divided into 7 Key factor Topics as follows:"}</h1>', unsafe_allow_html=True)

        col1 , col2 , col3 , col4 , col5 , col6 , col7= st.columns(7)
        col1.info('👀 Body / Design / Looks/ Style')
        col2.info('⚡ Engine / Performance / Speed')
        col3.info('🛠️ Service & Maintenance')
        col4.info('🆕 Special Feature, New feature')
        col5.info('🥇 Competition')
        col6.info('🏍️ Ride experience / Comfortability')
        col7.info('💲 Price, Cost, Buying')
    
        if product == 'Suzuki':
            st.write(f'<h1 style="margin-top:-20px;text-align: left;color:#9FACB8;font-size:25px;">{"Number of Topics per Key Factors for Suzuki"}</h1>', unsafe_allow_html=True)
            with open("suzuki_piechart.html", "r") as f:
                html_content1 = f.read()
        # components.v1.html(html_content, width=1200, height=3000, scrolling=True)

            with st.container():
            # st.write("Marketing Mix Tree")
            # Display the HTML content in the Streamlit app within the container
                components.v1.html(html_content1, height=500, scrolling=False)

            st.write(f'<h1 style="margin-top:-20px;text-align: left;color:#9FACB8;font-size:25px;">{"Automotive Key Factors Tree for Suzuki"}</h1>', unsafe_allow_html=True)
            with open("tree.html", "r") as f:
                html_content = f.read()
        # components.v1.html(html_content, width=1200, height=3000, scrolling=True)

            with st.container():
            # st.write("Marketing Mix Tree")
            # Display the HTML content in the Streamlit app within the container
                components.v1.html(html_content, height=800, scrolling=True)
                
        if product == 'Honda':
            st.write(f'<h1 style="margin-top:-20px;text-align: left;color:#9FACB8;font-size:25px;">{"Number of Topics per Key Factors for Honda"}</h1>', unsafe_allow_html=True)
            with open("honda_piechart.html", "r") as f:
                html_content1 = f.read()
        # components.v1.html(html_content, width=1200, height=3000, scrolling=True)

            with st.container():
            # st.write("Marketing Mix Tree")
            # Display the HTML content in the Streamlit app within the container
                components.v1.html(html_content1, height=500, scrolling=False)

            st.write(f'<h1 style="margin-top:-20px;text-align: left;color:#9FACB8;font-size:25px;">{"Automotive Key Factors Tree for Suzuki"}</h1>', unsafe_allow_html=True)
            with open("tree_honda.html", "r") as f:
                html_content = f.read()
        # components.v1.html(html_content, width=1200, height=3000, scrolling=True)

            with st.container():
            # st.write("Marketing Mix Tree")
            # Display the HTML content in the Streamlit app within the container
                components.v1.html(html_content, height=800, scrolling=True)
        if product == 'TVS':
            st.write(f'<h1 style="margin-top:-20px;text-align: left;color:#9FACB8;font-size:25px;">{"Number of Topics per Key Factors for TVS"}</h1>', unsafe_allow_html=True)
            with open("tvs_piechart.html", "r") as f:
                html_content1 = f.read()
        # components.v1.html(html_content, width=1200, height=3000, scrolling=True)

            with st.container():
            # st.write("Marketing Mix Tree")
            # Display the HTML content in the Streamlit app within the container
                components.v1.html(html_content1, height=500, scrolling=False)

            st.write(f'<h1 style="margin-top:-20px;text-align: left;color:#9FACB8;font-size:25px;">{"Automotive Key Factors Tree for Suzuki"}</h1>', unsafe_allow_html=True)
            with open("tree_tvs.html", "r") as f:
                html_content = f.read()
        # components.v1.html(html_content, width=1200, height=3000, scrolling=True)

            with st.container():
            # st.write("Marketing Mix Tree")
            # Display the HTML content in the Streamlit app within the container
                components.v1.html(html_content, height=800, scrolling=True)

elif tabs == 'Sentiment Analysis':
        sac.divider(label='💭', align='center')
        st.write(f'<h1 style="margin-top:-20px;color:#094780;font-size:30px;">{"Customer Sentiment"}</h1>', unsafe_allow_html=True)
        # st.title("Customer Sentiment")
        # st.write('Customer Sentiment'.format(tabs))
        st.write(f'<h1 style="margin-top:-20px;text-align: left;color:#9FACB8;font-size:20px;">{"Topic Insights For Brand: Likes & Dislikes"}</h1>', unsafe_allow_html=True)
        # if product == 'Suzuki':
        col1, col2 = st.columns(2)
        # with st.expander("Click to see insights"):
        if product == 'Suzuki':
                col1.write(f'<h1 style="margin-top:-20px;text-align: center;color:#9FACB8;font-size:20px;">{"Likes👍"}</h1>', unsafe_allow_html=True)
                col1.info('''
                - **Riding Experience and Road Conditions**: Riders appreciate the smooth handling and comfortable suspension on various road surfaces.
                - **Vehicle Buying Experience**: Positive feedback is received for dealerships with straightforward purchasing processes and friendly sales staff.
                - **Vehicle Maintenance and Component Considerations**: Owners like the durability of components and reasonable maintenance costs.
                - **Positive Vehicle Experience and Appreciation**: Customers express loyalty to the brand and share memorable riding experiences.
                - **Comfort and Quality of Scooter Seating, particularly for long rides**: Owners enjoy comfortable seating and ergonomic design for extended journeys.''')

                # Customer preferences
                col2.write(f'<h1 style="margin-top:-20px;text-align: center;color:#9FACB8;font-size:20px;">{"Dislikes👎"}</h1>', unsafe_allow_html=True)
                col2.info('''
                - **Vehicle Starting Issues and Engine Problems**: Some users report occasional starting problems and engine issues, which can be frustrating.
                - **Vehicle Body and Design Considerations**: There are dislikes related to limited color options and outdated design in certain models.
                - **Mileage and Scooter Comparison, with emphasis on Honda Activa**: Some customers are disappointed by lower-than-expected mileage and unfavorable comparisons with Honda Activa.
                - **Issues and Experiences with Scooter Service and Performance**: Negative experiences are reported, including frequent service visits and unresolved problems.
                - **Features and Buying Considerations**: Some buyers express disappointment with limited feature choices and uninformed purchases.''')
                
                st.write(f'<h1 style="margin-top:-20px;text-align: left;color:#9FACB8;font-size:20px;">{"Key Factors Insights"}</h1>', unsafe_allow_html=True)
                pb1, pb2, pb3, pb4,pb5,pb6,pb7 = st.tabs(["Body / Design / Looks/ Style","Engine / Performance / Speed","Service & Maintenance","Special Feature, New feature","Competittion","Ride experience / Comfortability","Price, Cost, Buying"])

                with pb1:
            # st.title("Body")
                    st.write("""
                    - Keywords: grip, shock, sound, appeal, quality, body, space, small, helmet, look, light, head, chrome, colour.
                    - Insights:
                        1. Customers are concerned about the grip of the scooter's tires and the shock absorption.
                        2. Aesthetic factors such as appeal, quality, and looks play a significant role in the scooter's choice.
                        3. Space and design, including small details like helmet hooks and lighting, are important to buyers.
                    """)

                with pb2:
            # st.title("Engine")
                    st.write("""
                   - Keywords: speed, engine, performance, power, cc, tank, consumption, mileage, pickup, speedometer, weight, handle.
                   - Insights:
                       1. Performance-related factors like speed, engine power, and fuel efficiency are essential considerations.
                       2. Mileage, pickup, and handling are significant aspects of the scooter's performance.
                       3. Engine specifications such as cc, tank capacity, and fuel consumption are mentioned.
                    """)

                with pb3:
            # st.title("service")
                    st.write("""
                    - Keywords: service, part, center, oil, servicing, spare, problem, issue, battery, problem, kms, face, review, company.
                    - Insights:
                            1. Customers are concerned about the scooter's maintenance and service requirements.
                            2. Availability of spare parts, servicing centers, and addressing issues are important considerations.
                            3. Mileage and performance issues are discussed in the context of service and maintenance.
                    """)
            
                with pb4:
            # st.title("special feature")
                    st.write("""
                    - Keywords: special, edition, key, cover, open, lock, digital meter, make, new, model, india.
                    - Insights:
                    1. Customers are interested in special editions and new features like digital meters and keyless entry.
                    2. The availability of unique features can influence purchasing decisions.
                    """)

                with pb5:
            # st.title("competition")
                    st.write("""
                    - Keywords: vehicle, dealer, purchase, compare, access, honda, suzuki, friend, activa, scooty, company, activa, inda.
                    - Insights:
                        1. Customers often compare different scooter models, including Honda Activa, Suzuki Access, and others.
                        2. Recommendations from friends and the reputation of the company are mentioned in the context of competition.
                    """)
            
                with pb6:
                    # st.title("ride experience")
                    st.write("""
                    - Keywords: ride, comfortable, comfort, seat, long, space, drive, sit, feel, test, people, smooth.
                    - Insights:
                        1. Comfort during rides, including seat comfort and spaciousness, is crucial to buyers.
                        2. Test rides and user experiences play a significant role in evaluating comfort.
                        3. The smoothness of the ride is an important factor.
                    """)

                with pb7:
            # st.title("price")
                    st.write("""
                    - Keywords: buy, purchase, price, cost.
                    - Insights:
                        1. Pricing and affordability are essential considerations for potential buyers.
                        2. The cost of purchasing the scooter is a key decision-making factor.""")
                    
        if product == 'Honda':
                col1.write(f'<h1 style="margin-top:-20px;text-align: center;color:#9FACB8;font-size:20px;">{"Likes👍"}</h1>', unsafe_allow_html=True)
                col1.info('''
                - **Comparing the Honda Activa's Engine and Riding Experience**: Enthusiasts appreciate the comparison of engine performance and riding experience, seeking information to make informed choices.
                - **Exploring the Best Features and Colors for an Awesome and Comfortable Ride**: Riders value information on features and colors that enhance the comfort and enjoyment of their motorcycle experience.
                - **Maximizing Mileage and Performance**: Riders seek tips on improving engine efficiency, lighting, and smooth riding to enhance their overall motorcycle experience.
                - **Awesome Look and Affordable Price Range of Honda Motorcycles**: Buyers appreciate the combination of attractive design and affordability in Honda motorcycles.
                - **Choosing the Perfect Scooty**: Prospective buyers are interested in factors to consider when making a scooter purchase, indicating a desire for informed decision-making. ''')

                # Customer preferences
                col2.write(f'<h1 style="margin-top:-20px;text-align: center;color:#9FACB8;font-size:20px;">{"Dislikes👎"}</h1>', unsafe_allow_html=True)
                col2.info('''
                - **Honda Scooter Ownership**: Some riders may face challenges and common problems with Honda scooters, suggesting potential negative experiences.
                - **Powerful and Attractive Commuter Bikes**: There may be concerns or dislikes related to commuter bikes in terms of performance or styling.
                - **Troubleshooting Common Motorcycle Problems**: Riders encounter issues related to mileage, speed, and engine performance, which can be seen as negative experiences.
                - **Optimizing Your Honda Motorcycle Service Experience**: Riders may have concerns about managing service costs and maintenance, potentially reflecting negative aspects of the ownership experience.
                - **Honda Motorcycle Gear Shift Issues**: Some riders may experience gear shift problems, impacting the smoothness of their rides.''')
                
                st.write(f'<h1 style="margin-top:-20px;text-align: left;color:#9FACB8;font-size:20px;">{"Key Factors Insights"}</h1>', unsafe_allow_html=True)
                pb1, pb2, pb3, pb4,pb5,pb6,pb7 = st.tabs(["Body / Design / Looks/ Style","Engine / Performance / Speed","Service & Maintenance","Special Feature, New feature","Competittion","Ride experience / Comfortability","Price, Cost, Buying"])

                with pb1:
            # st.title("Body")
                    st.write("""
                    - Keywords: look, feature, colour, light, awesome, smooth.
                    - Insights: 
                        1. Customers are discussing the scooter's look, color options, and design features. They appreciate features that enhance the scooter's style and appearance.
                    """)

                with pb2:
            # st.title("Engine")
                    st.write("""
                    - Keywords: engine, power, speed, mileage, performance, shift.
                    - Insights: 
                        1. Customers are sharing their experiences related to the scooter's engine performance, power, speed, and fuel efficiency. Some mention gear shift performance.
                    """)

                with pb3:
            # st.title("service")
                    st.write("""
                    - Keywords: service, problem, issue, maintenance, cost.
                    - Insights: 
                        1. Customers are discussing service-related aspects, including problems, maintenance, and associated costs. They are concerned about service quality.
                    """)

                with pb4:
            # st.title("special feature")
                    st.write("""
                    - Keywords: feature, headlight.
                    - Insights: 
                        1. Customers are talking about special features, especially regarding headlight performance.
                    """)

                with pb5:
            # st.title("competition")
                    st.write("""
                    - Keywords: activa, honda, scooty, xblade.
                    - Insights: 
                        1. Customers are comparing the scooter with Honda Activa, Scooty, and Xblade in terms of various aspects.
                    """)
            
                with pb6:
                    # st.title("ride experience")
                    st.write("""
                    - Keywords: ride, comfortable, feel, long, time.
                    - Insights: 
                        1. Customers are sharing their riding experiences and comfort-related feedback, including long rides.
                    """)

                with pb7:
            # st.title("price")
                    st.write("""
                    - Keywords: buy, purchase, cost, value, money.
                    - Insights: 
                        1. Customers are discussing the price, cost, and overall value for money when considering buying a scooter.""")
                    
        if product == 'TVS':
                col1.write(f'<h1 style="margin-top:-20px;text-align: center;color:#9FACB8;font-size:20px;">{"Likes👍"}</h1>', unsafe_allow_html=True)
                col1.info('''
                - **Maximizing the Value of Your Motorcycle**: Riders appreciate motorcycles that offer superb power, great rides, and smart money choices.
                - **Indian Motorcycle Machines**: Enthusiasts enjoy the journey of feel, the competitive pricing, and the riding pleasure that Indian motorcycles provide.
                - **Striking the Perfect Balance in Motorcycle Ownership**: Riders value motorcycles that offer good rides, ideal prices, and strong performance, indicating a desire for well-rounded options.
                - **Unleashing Awesome Rides and Sporty Looks**: Motorcycle enthusiasts favor motorcycles that offer exciting rides and sporty aesthetics.
                - **The Scooter Revolution**: Riders appreciate fuel efficiency, stylish rides, and positive rider experiences when it comes to scooters.''')

                # Customer preferences
                col2.write(f'<h1 style="margin-top:-20px;text-align: center;color:#9FACB8;font-size:20px;">{"Dislikes👎"}</h1>', unsafe_allow_html=True)
                col2.info('''
                - **Navigating Service Challenges and Making Informed Buying Decisions**: Some riders express frustration with service challenges, and there is a need for better information to make informed buying decisions.
                - **Troubleshooting Vehicle Issues, Starting Strong, and Self-Improvement**: Riders face challenges with troubleshooting vehicle issues and starting problems, which can be seen as negative experiences.
                - **The Art of Riding**: There are concerns or dislikes related to aspects of motorcycle aesthetics or style.
                - **Elevating the Scooter Experience**: Some riders may have reservations about the scooter experience, possibly due to expectations of greater innovation.
                - **Navigating the World of Motorcycle Ownership**: Negative experiences are reported, including challenges with service, maintenance, and the overall rider's journey.''')
               
                st.write(f'<h1 style="margin-top:-20px;text-align: center;color:#9FACB8;font-size:20px;">{"Key Factors Insights"}</h1>', unsafe_allow_html=True)
                pb1, pb2, pb3, pb4,pb5,pb6,pb7 = st.tabs(["Body / Design / Looks/ Style","Engine / Performance / Speed","Service & Maintenance","Special Feature, New feature","Competittion","Ride experience / Comfortability","Price, Cost, Buying"])

                with pb1:
            # st.title("Body")
                    st.write("""
                   - Keywords: Look, style, design, nice, comfortable, new
                   - Insights: 
                        1. Customers frequently comment on the scooter's appearance, highlighting its nice design and comfortable style. Many find it visually appealing and consider it a new and attractive option in terms of style.
                    """)

                with pb2:
            # st.title("Engine")
                    st.write("""
                   - Keywords: Power, speed, engine, performance, pickup, mileage
                   - Insights: 
                        1. Customers focus on the scooter's performance attributes, emphasizing factors like power, speed, engine performance, and pickup. Mileage is also discussed, indicating an interest in fuel efficiency.
                    """)

                with pb3:
            # st.title("service")
                    st.write("""
                    - Keywords: Service, maintenance, problem, issue
                    - Insights: 
                        1. Customers share their experiences with the scooter's service and maintenance. They discuss any problems or issues they've encountered, providing feedback on the scooter's reliability and after-sales service.
                    """)

                with pb4:
            # st.title("special feature")
                    st.write("""
                    - Keywords: Feature, awesome, great, useful
                    - Insights: 
                        1. Customers appreciate unique features in the scooter and describe them as awesome and great. They value features that enhance their riding experience, emphasizing their usefulness.
                    """)

                with pb5:
            # st.title("competition")
                    st.write("""
                    - Keywords: Jupiter, TVS, segment, class, braking
                    - Insights: 
                        1. Customers compare the scooter to competitors like Jupiter and other TVS models. They evaluate its performance and class in relation to these competitors and mention considerations like braking.
                    """)
            
                with pb6:
                    # st.title("ride experience")
                    st.write("""
                    - Keywords: Comfortable, smooth, cost, light
                    - Insights: 
                           1. Customers focus on the riding experience, emphasizing comfort and smoothness. They also discuss cost-related aspects and the scooter's lightweight nature.
                    """)

                with pb7:
            # st.title("price")
                    st.write("""
                    - Keywords: Price, value, money, buying, purchase
                    - Insights: 
                        1. Customers discuss the scooter's price, value for money, and their buying experiences. They evaluate whether the scooter is a worthwhile purchase.
                    - Additional Insights:
                        1. This category includes various keywords and insights that don't fit into the predefined categories. These insights cover a range of topics, including specific scooter models (e.g., "Apache," "RR"), mentions of "excellent" features, and discussions about "smooth" rides. These insights provide additional context and feedback from customers.
                    """)
                    
elif tabs == 'Competitive Analysis':
        sac.divider(label='🏆', align='center')
        # st.title("Competition")
        st.write(f'<h1 style="margin-top:-20px;color:#094780;font-size:30px;">{"Competition"}</h1>', unsafe_allow_html=True)
        st.info('This is an analysis of customer reviews across bike segment of its competitors.')
        st.write(f'<h1 style="margin-top:-15px;color:#094780;font-size:15px;">{"Total Data collected for different Brands"}</h1>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
    
        col1.image('suzuki_logo.png',width=200)
    # col11.metric("Amazon", Amazon_count)
        col2.image('Honda_logo.png',width=200)
    # col22.metric("Wallmart", Wallmart_count)
        col3.image('tvs_logo.png', width=200)
        coll1, coll2, coll3 = st.columns(3)
        coll1.metric("Suzuki", **7385**)
        coll2.metric("Honda", 1846)
        coll3.metric("TVS", 1033)
        st.write(f'<h1 style="margin-top:-20px;text-align: left;color:#9FACB8;font-size:20px;">{"Mean Polarity across Brands"}</h1>', unsafe_allow_html=True)
        st.info("Percentage of Positive, Neutral and Negative feedback across different products. We have selected Positive reviews with polarity score > 0.4, Neutral reviews within 0.2 to 0.4 & Negative reviews below 0.2 (Polarity scores are calculated using machine learning for sentiment analysis on text data. It ranges between -1 to 1). ")
        col1, col2, col3 = st.columns(3)
        lst = ['Green', 'Orange', 'Red']
        with col1:
            st.subheader("Suzuki")
            emo_dict = {
                "Positive": str(63.78) + "%",
                "Neutral": str(15.75) + "%",
                "Negative": str(20.47) + "%"
            }
            metrics = list(emo_dict.items())

        # Display each metric using st.metric
            for metric_key, metric_value in emo_dict.items():
                if metric_key == 'Negative':
                    metric_value = f'<span style="color:red;">{metric_value}</span>'
                elif metric_key == 'Positive':
                    metric_value = f'<span style="color:green;">{metric_value}</span>'
                elif metric_key == 'Neutral':
                    metric_value = f'<span style="color:orange;">{metric_value}</span>'
                st.markdown(f'{metric_key}: {metric_value}', unsafe_allow_html=True)

        with col2:
            st.subheader("Honda")
            emo_dict = {
                "Positive": str(80.85) + "%",
                "Neutral": str(7.80) + "%",
                "Negative": str(11.35) + "%"
            }
            metrics = list(emo_dict.items())

            for metric_key, metric_value in emo_dict.items():
                if metric_key == 'Negative':
                    metric_value = f'<span style="color:red;">{metric_value}</span>'
                elif metric_key == 'Positive':
                    metric_value = f'<span style="color:green;">{metric_value}</span>'
                elif metric_key == 'Neutral':
                    metric_value = f'<span style="color:orange;">{metric_value}</span>'
                st.markdown(f'{metric_key}: {metric_value}', unsafe_allow_html=True)

        with col3:
            st.subheader("TVS")
            emo_dict = {
                "Positive": str(79.765) + "%",
                "Neutral": str(13.35) + "%",
                "Negative": str(6.87) + "%"
            }
            metrics = list(emo_dict.items())

            for metric_key, metric_value in emo_dict.items():
                if metric_key == 'Negative':
                    metric_value = f'<span style="color:red;">{metric_value}</span>'
                elif metric_key == 'Positive':
                    metric_value = f'<span style="color:green;">{metric_value}</span>'
                elif metric_key == 'Neutral':
                    metric_value = f'<span style="color:orange;">{metric_value}</span>'
                st.markdown(f'{metric_key}: {metric_value}', unsafe_allow_html=True)
        st.write(f'<h1 style="margin-top:-20px;text-align: left;color:#9FACB8;font-size:20px;">{"Overall Polarity :"}</h1>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Suzuki", 0.392)
        col2.metric("Honda", 0.618)
        col3.metric("TVS", 0.519)
    # col4.metric("K", scores['Polarity Scores'][3])

        st.write(f'<h1 style="margin-top:-15px;color:#094780;font-size:15px;">{"Polarity across different Subtopics"}</h1>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        col1.metric("Overall Polarity","")
        dff = pd.read_csv('mean_polarity.csv')
        df2 = dff[['Brand','Overall Polarity']]
        df2['Overall Polarity']=df2['Overall Polarity'].astype('float')
        col1.bar_chart(df2, x= "Brand",y="Overall Polarity")

        # col1.image('Key_Factor0.png',width=500)
        # col1.metric("Overall Polarity","")
        # col11.metric("Amazon", Amazon_count)
        col2.metric("Body/Design/Looks/Style","")
        df2 = dff[['Brand','Body/Design/Looks/Style']]
        df2['Body/Design/Looks/Style']=df2['Body/Design/Looks/Style'].astype('float')
        col2.bar_chart(df2, x= "Brand",y="Body/Design/Looks/Style")
    
        # col2.image('Key_Factor1.png',width=500)
        coll1, coll2 = st.columns(2)
        coll1.metric("Engine/Performance/Speed","")
        df2 = dff[['Brand','Engine/Performance/Speed']]
        df2['Engine/Performance/Speed']=df2['Engine/Performance/Speed'].astype('float')
        coll1.bar_chart(df2, x= "Brand",y="Engine/Performance/Speed")
    
        # coll1.image('Key_Factor2.png',width=500)
    # col11.metric("Amazon", Amazon_count)
        coll2.metric("Service & Maintenance","")
        df2 = dff[['Brand','Service & Maintenance']]
        df2['Service & Maintenance']=df2['Service & Maintenance'].astype('float')
        coll2.bar_chart(df2, x= "Brand",y="Service & Maintenance")
    
        # coll2.image('Key_Factor3.png',width=500)
        colll1, colll2 = st.columns(2)
        colll1.metric("Special Feature, New Feature","")
        df2 = dff[['Brand','Special Feature, New Feature']]
        df2['Special Feature, New Feature']=df2['Special Feature, New Feature'].astype('float')
        colll1.bar_chart(df2, x= "Brand",y="Special Feature, New Feature")
    
        # colll1.image('Key_Factor4.png',width=500)
    # col11.metric("Amazon", Amazon_count)
        colll2.metric("Competition","")
        df2 = dff[['Brand','Competition']]
        df2['Competition']=df2['Competition'].astype('float')
        colll2.bar_chart(df2, x= "Brand",y="Competition")
    
        # colll2.image('Key_Factor5.png',width=500)
        collll1, collll2 = st.columns(2)
        collll1.metric("Ride Experience/Comfortability","")
        df2 = dff[['Brand','Ride Experience/Comfortability']]
        df2['Ride Experience/Comfortability']=df2['Ride Experience/Comfortability'].astype('float')
        collll1.bar_chart(df2, x= "Brand",y="Ride Experience/Comfortability")
    
        # collll1.image('Key_Factor6.png',width=500)
    # col11.metric("Amazon", Amazon_count)
        collll2.metric("Price, Cost, Buying","")
        df2 = dff[['Brand','Price, Cost, Buying']]
        df2['Price, Cost, Buying']=df2['Price, Cost, Buying'].astype('float')
        collll2.bar_chart(df2, x= "Brand",y="Price, Cost, Buying")
        # collll2.image('Key_Factor7.png',width=500)
