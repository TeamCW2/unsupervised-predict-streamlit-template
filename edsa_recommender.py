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
from select import select
from tkinter import HORIZONTAL
import streamlit as st  
from streamlit_option_menu import option_menu

# Data handling dependencies
import pandas as pd
import numpy as np

# Custom Libraries
from utils.data_loader import load_movie_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model
from PIL import Image

# Data Loading
title_list = load_movie_titles('resources/data/movies.csv')

# App declaration
def main():


    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.

    st.sidebar.image('resources/imgs/Disneylogo.png', use_column_width=True) 
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
            sys = st.radio("Select an algorithm",
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
                                
# -------------------------------------------------------------------
 # ------------- SAFE FOR ALTERING/EXTENSION -------------------
                                
    if selected == 'Exploratory Data Analysis':
        st.title('Exploratory Data Analysis')
        st.image('resources/imgs/AI-new-1.jpg',use_column_width=True) 
        if st.checkbox("ratings"):
            st.subheader("1. Movie ratings")
            st.image('resources/imgs/distribution-ratings.png',use_column_width=True)
            st.subheader('2. Compared to movies that have lower average ratings, movies with higher average ratings actually have more number of ratings') 
            st.image('resources/imgs/Ratings.png',use_column_width=True)

        if st.checkbox("genre wordcloud"):
            st.subheader("Top Genres")
            st.write('A worldcloud of the different most searched genres')
            st.image('resources/imgs/wordcloud.png',use_column_width=True)
            
        #if st.checkbox("genres"):
            #st.subheader("Top Genres")
            #st.image('resources/imgs/top_genres.PNG',use_column_width=True)

        if st.checkbox("Relevance"):
            st.subheader("Data taken from genome_scores")
            st.image('resources/imgs/boxplot.png',use_column_width=True)

        if st.checkbox("movies"):
            st.subheader("avarage rating per director")
            st.image('resources/imgs/visual2.png',use_column_width=True)
            st.subheader("movies with 100 or more viewers")
            st.image('resources/imgs/visual1.png',use_column_width=True)

        

        # ------------- SAFE FOR ALTERING/EXTENSION -------------------
    if selected == "Solution Overview":
        st.title("Solution Overview")
            #st.write("RMSE of the recommendation models to show their performance")
        st.markdown('We employed two methods of building recommmendation system:')
        st.markdown('1. Content-based filtering')
        st.markdown('2. Collaborative filtering')
        st.markdown("The Content-Based Recommendation system computes similarity between movies based on movie genres using the selected movie as a baseline.This system uses item metadata, such as genre, director, description, actors, etc. for movies, to make these recommendations. The general idea behind these recommender systems is that if a person likes a particular movie, he or she will also like a movie that is similar to it. And to recommend that, it will make use of the user's past movie metadata..Let's create functions which operate those functions.")
        st.markdown("Collaborative methods for recommender systems are methods that are based solely on the past interactions recorded between users and items in order to produce new recommendations. These methods do not require item meta-data like their content-based counterparts. This makes them less memory intensive which is a major plus since our dataset is so huge.")
        st.markdown("Our best perfoming solution to a movie recommender sytem was the collaborative filtering. We intergrated it with our best perfoming model which is the SVD model as seen in the graph below.")
        st.subheader('A graph of Model Perfomances')
        st.image('resources/imgs/models.png',use_column_width=True)
        st.markdown('From the image above the SVD model perfomed best with an RMSE of 0.834903 as compared to the other models.')
        st.markdown('The singular value decomposition (SVD) provides another way to factorize a matrix, into singular vectors and singular values. The SVD allows us to discover some of the same kind of information as the eigen decomposition.The SVD is used widely both in the calculation of other matrix operations, such as matrix inverse, but also as a data reduction method in machine learning. SVD can also be used in least squares linear regression, image compression, and denoising data.')

    if selected == "About":  
        st.write("### Overview: Generating a movie recommender system using unsupervised learning skills")
        st.image('resources/imgs/popcorn2-new-1.jpg',use_column_width=True)   
        # You can read a markdown file from supporting resources folder

        st.subheader("Introduction to Unsupervised Learning Predict")
        st.write("""Recommender systems are among the most popular applications of data science today. They are used to predict the "rating" or "preference" that a user would give to an item. Almost every major tech company has applied them in some form. One application where this is especially true surrounds movie content recommendations; where intelligent algorithms can help viewers find great titles from tens of thousands of options.""")
        st.write("""With this context, EDSA is challenging you to construct a recommendation algorithm based on content or collaborative filtering, capable of accurately predicting how a user will rate a movie they have not yet viewed based on their historical preferences.""")
        st.write("""Providing an accurate and robust solution to this challenge has immense economic potential, with users of the system being exposed to content they would like to view or purchase - generating revenue and platform affinity.""")

        
        st.subheader("Problem Statement of the Unsupervised Learning Predict")
        st.write("In this machine learning project, we build movie recommendation systems. We built a content-based recommendation engine that makes recommendations given the title of the movie as input.")

        #now we are looking at the data 
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
        st.header("Help Desk")
        st.subheader("Please give us feedback about this app")
        st.write("""Please use the reaching out options below to provide feedback on content and/or your experience using this app. We're always looking to improve, so we appreciate hearing from you! Though we can't respond to submissions individually, our teams do review all feedback. Please do not include any personal information.""")
        
        st.set_page_icon:('resources/imgs/Logo.JPEG')

        
        sys=st.selectbox("Communication Preference",
                    ("Email",
                    "WhatsApp",
                    "Live Chat Support"))
        if sys == 'Email':
            
        # You can read a markdown file from supporting resources folder
        
            st.markdown("""
            
        6 data scientists and developers were assinged to this project from Data Avangers company.
        You can reach out to anyone of them through the provided Emails regarding any query : """) 
            
            
            st.write('Mapula : maponyavictoria@gmail.com')
            st.write('Mbali  :mbali.mnguni6@gmail.com ')
            st.write('Immaculate : immaculatemakokga@gmail.com')
            st.write('Lesego :lesegophaahla97@gmail.com')
            st.write('Mxolisi :mxolisiaubreykhumalo@gmail.com')

        if sys == 'WhatsApp':
            st.write("Updates and special offers")
            st.write("Quick Tip – For fastest resolution to your queries/ issues, make sure you have access to your subscribed mobile number before you start chatting with us.")
            st.image('resources/imgs/W-backround.png')
            st.button("0762358899")
        

        if sys == 'Live Chat Support':
            st.write("Our offices are currently closed. If you are unable to find a solution to your query/issue on our Help Center, please reach out to us tomorrow and we will assist you. Thank you.")
            st.image('resources/imgs/disappointed-expression-1-new.png')



        
        
        st.subheader("Need more help?")
        st.write("Call Us On:")
        st.button("01125693289")
        st.write("Call us from your subscribed mobile number.")
        st.image('resources/imgs/Logo3.jpeg')
    
if __name__ == '__main__':
    main()

