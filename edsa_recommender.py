"""

    Streamlit webserver-based Recommender Engine.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: !! Do not remove/modify the code delimited by dashes !!

    This application is intended to be partly marked in an automated manner.
    Altering delimited code may result in a mark of 0.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend certain aspects of this script
    and its dependencies as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st  
from streamlit_option_menu import option_menu

# Data handling dependencies
import pandas as pd
import numpy as np

# Custom Libraries
from utils.data_loader import load_movie_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model


# Data Loading
title_list = load_movie_titles('resources/data/movies.csv')

# App declaration


    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.

		#st.sidebar.image(Image.open('resources/imgs/Logo3.png'))
with st.sidebar:
    selected = option_menu(
			menu_title = 'Choose option',
			menu_icon="list", 
			options = ['Recommender System','Exploratory Data Analysis',"Solution Overview",'About',
            'Help center'],
			)

    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # ------------------------------------------------------------------- 
if selected == "Recommender System":
        # Header contents
        st.write('# Movie Recommender Engine')
        st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        st.image('resources/imgs/Image_header.png',use_column_width=True)
        # Recommender System algorithm selection
        sys = st.selectbox("Select an algorithm",
                       ('Content Based Filtering',
                        'Collaborative Based Filtering'))

        # User-based preferences
        st.write('### Enter Your Three Favorite Movies')
        movie_1 = st.selectbox('First Option',title_list[14930:15200])
        movie_2 = st.selectbox('Second Option',title_list[25055:25255])
        movie_3 = st.selectbox('Third Option',title_list[21100:21200])
        fav_movies = [movie_1,movie_2,movie_3]

        # Perform top-10 movie recommendation generation
        if sys == 'Content Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = content_model(movie_list=fav_movies,
                                                            top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


        if sys == 'Collaborative Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = collab_model(movie_list=fav_movies,
                                                           top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")
                              
if selected == 'Exploratory Data Analysis':
    st.title('Exploratory Data Analysis')
    if st.checkbox("ratings"):
        st.subheader("Movie ratings")
        st.image('resources/imgs/rating.PNG',use_column_width=True)
        
    if st.checkbox("genre wordcloud"):
        st.subheader("Top Genres")
        st.image('resources/imgs/genre_wordcloud.png',use_column_width=True)
        
    if st.checkbox("genres"):
        st.subheader("Top Genres")
        st.image('resources/imgs/top_genres.PNG',use_column_width=True)

    if st.checkbox("tags"):
        st.subheader("Top tags")
        st.image('resources/imgs/top_tags.PNG',use_column_width=True)

    if st.checkbox("cast"):
        st.subheader("Popular cast")
        st.image('resources/imgs/cast.PNG',use_column_width=True)


    # ------------- SAFE FOR ALTERING/EXTENSION -------------------
if selected == "Solution Overview":
    st.title("Solution Overview")
         #st.write("RMSE of the recommendation models to show their performance")
    st.markdown('We employed two methods of building recommmendation system:')
    st.markdown('1. Content-based filtering')
    st.markdown('2. Collaborative filtering')
    st.markdown("The Content-Based Recommendation system computes similarity between movies based on movie genres using the selected movie as a baseline. Using this type of movie recommendation system, we require the title of the movie as input, but our sim_matrix is based on the index of each movie. Therefore, to build this, we need to convert movie title into movie index and movie index into movie title. Let's create functions which operate those functions.")
    st.markdown("Collaborative methods for recommender systems are methods that are based solely on the past interactions recorded between users and items in order to produce new recommendations. These methods do not require item meta-data like their content-based counterparts. This makes them less memory intensive which is a major plus since our dataset is so huge.")
    st.markdown("Our best perfoming solution to a movie recommender sytem was the collaborative filtering. We intergrated it with our best perfoming model which is the SVD model as seen in the graph below.")
    st.subheader('A graph of Model Perfomances')
    st.image('resources/imgs/Models.jpeg',use_column_width=True)
    st.markdown('From the image above the SVD model perfomed best with an RMSE of 0.906 as compared to the other models.')
    st.markdown('The singular value decomposition (SVD) provides another way to factorize a matrix, into singular vectors and singular values. The SVD allows us to discover some of the same kind of information as the eigen decomposition.The SVD is used widely both in the calculation of other matrix operations, such as matrix inverse, but also as a data reduction method in machine learning. SVD can also be used in least squares linear regression, image compression, and denoising data.')

if selected == "About":  
    st.write("### Oveview: Flex your Unsupervised Learning skills to generate movie recommendations")
        
    # You can read a markdown file from supporting resources folder
    #if st.checkbox("Introduction"):
    st.subheader("Introduction to Unsupervised Learning Predict")
    st.write("""In today’s technology driven world, recommender systems are socially and economically critical for ensuring that individuals can make appropriate choices surrounding the content they engage with on a daily basis. One application where this is especially true surrounds movie content recommendations; where intelligent algorithms can help viewers find great titles from tens of thousands of options.""")
    st.write("""With this context, EDSA is challenging you to construct a recommendation algorithm based on content or collaborative filtering, capable of accurately predicting how a user will rate a movie they have not yet viewed based on their historical preferences.""")
    st.write("""Providing an accurate and robust solution to this challenge has immense economic potential, with users of the system being exposed to content they would like to view or purchase - generating revenue and platform affinity.""")

    #if st.checkbox("Problem Statement"):
    st.subheader("Problem Statement of the Unsupervised Learning Predict")
    st.write("Build recommender systems to recommend a movie")

    #if st.checkbox("Data"):
    st.subheader("Data Overview")
    st.write("""This dataset consists of several million 5-star ratings obtained from users of the online MovieLens movie recommendation service. The MovieLens dataset has long been used by industry and academic researchers to improve the performance of explicitly-based recommender systems, and now you get to as well!""")

    st.write("""For this Predict, we'll be using a special version of the MovieLens dataset which has enriched with additional data, and resampled for fair evaluation purposes.""")

    st.write("""### Source:""") 
    st.write("""The data for the MovieLens dataset is maintained by the GroupLens research group in the Department of Computer Science and Engineering at the University of Minnesota. Additional movie content data was legally scraped from IMDB""")


    st.write("""### Supplied Files:
    genome_scores.csv - a score mapping the strength between movies and tag-related properties. Read more here

    genome_tags.csv - user assigned tags for genome-related scores

    imdb_data.csv - Additional movie metadata scraped from IMDB using the links.csv file.

    links.csv - File providing a mapping between a MovieLens ID and associated IMDB and TMDB IDs.

    sample_submission.csv - Sample of the submission format for the hackathon.

    tags.csv - User assigned for the movies within the dataset.

    test.csv - The test split of the dataset. Contains user and movie IDs with no rating data.

    train.csv - The training split of the dataset. Contains user and movie IDs with associated rating data.""")

   # You may want to add more sections here for aspects such as an EDA,
if selected == "Help center":
    st.subheader("Meet the Team")
        
        
    #st.image('resources/imgs/company.jpg', caption="Photo Credit: Hello I'm, AI Inc.com")

	# You can read a markdown file from supporting resources folder
    st.markdown("""
		
	Our team consists of 5 talented data scientists and developers from various parts of Africa.
    You can reach out to anyone of them through the provided mails: """) 
        
        
    st.write('Mapula Maponya: maponyavictoria@gmail.com')
    st.write('Mbhali  :mbali.mnguni6@gmail.com ')
    st.write('Immaculate : immaculatemakokga@gmail.com')
    st.write('Lesego :lesegophaahla97@gmail.com')
    st.write('Mxolisi :mxolisiaubreykhumalo@gmail.com')
    # or to provide your business pitch.

    
