import json
import pickle
from twitter import TwitterApp
import torch 
import pandas as pd
from io import BytesIO
import altair as alt
import pydeck as pdk
import streamlit as st 
from datetime import datetime, timedelta
from aws import aws_lambda, bucket

tw = TwitterApp()

@st.cache(show_spinner=False)
def get_prediction(text):
    event = json.dumps({'text': text})
    pred_response = aws_lambda.invoke(
        FunctionName='predict-dev-predict',
        InvocationType='Event',
        Payload=event
    )
    if pred_response['status'] == 200:
        return pred_response['prediction']
    else:
        return 'unknown'

@st.cache(show_spinner=False)
def get_hashtag_data(input, search_type, date_range):
    with st.spinner('gathering tweets...'):
        hashtags = tw.get_hashtag(input, search_type, date_range)
    with st.spinner('making predictions...'):
        for hashtag in hashtags:
            hashtag['sentiment'] = get_prediction(hashtag['text'])
        hashtag_data  = pd.DataFrame(hashtags)
        # hashtag_data['sentiment'] = hashtag_data.apply(lambda row: get_prediction(row.text), axis=1)
        # hashtag_data['sentiment'] = hashtag_data['sentiment'].replace(0, -1)
        # hashtag_data['created_at'] = hashtag_data.apply(lambda row: datetime.strptime(row.created_at, "%a %b %d %H:%M:%S %z %Y"), axis=1)
    return hashtag_data

st.title('Tweet sentiment analyzer by hashtag')

sidebar = st.sidebar

hashtag_input = sidebar.text_input('What hashtag would you like to search? ', 'covid19')
search_type = sidebar.selectbox('Select the type of search you want to do', ['popular', 'recent'], index=0)
if not hashtag_input.startswith('#'):
    hashtag_input = '#' + hashtag_input
hashtag_input = hashtag_input.replace(' ', '')

today = datetime.now().date()
week_ago =  today - timedelta(days=7)
date_range = sidebar.slider("How far back do you want to search? (Max 1 week)", min_value=week_ago, max_value=today, value=(week_ago, today), step=timedelta(days=1))

# 'Get the sentiment of the most {} tweets using {}'
button = st.button('Analyze Tweets!'.format(search_type, hashtag_input))

if button:
    hashtag_data = get_hashtag_data(hashtag_input, search_type, date_range)
    hashtag_data
    # hashtag_data = pd.read_excel(r'C:\Users\kevin.mcgee\Desktop\tweets.xlsx')
    # hashtag_data_date = hashtag_data.resample('H', on='created_at').sum().reset_index()
    # hashtag_data_date
    # chart = alt.Chart(hashtag_data_date).mark_line().encode(x='created_at', y='sentiment')
    # st.altair_chart(chart, use_container_width=True)
    
    # print(hashtag_data['geo'].value_counts())
    # hashtag_data[pd.notna(hashtag_data['geo'])]

#     st.pydeck_chart(pdk.Deck(
#      map_style='mapbox://styles/mapbox/light-v9',
#      initial_view_state=pdk.ViewState(
#          latitude=37.76,
#          longitude=-122.4,
#          zoom=11,
#          pitch=50,
#      ),
#      layers=[
#          pdk.Layer(
#             'HexagonLayer',
#             data=hashtag_data,
#             get_position='[lon, lat]',
#             radius=200,
#             elevation_scale=4,
#             elevation_range=[0, 1000],
#             pickable=True,
#             extruded=True,
#          ),
#          pdk.Layer(
#              'ScatterplotLayer',
#              data=df,
#              get_position='[lon, lat]',
#              get_color='[200, 30, 0, 160]',
#              get_radius=200,
#          ),
#      ],
#  ))