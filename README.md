# NLP Projects
This repository contains an overview of the content I learned while auditing CS124, "From Languages to Information" at Stanford University. It also contains my code for the final project for this class, when I implemented a movie recommendation Chatbot (The code is available upon request). 

## The MovieBot's functionalities 

The aim is to build a chatbot somewhat similar to [ELIZA](https://en.wikipedia.org/wiki/ELIZA), which would specifically interact with the user about his movie tastes, use sentiment analysis to extract sentiment associated with a set of movies, then recommend 10 different users accordingly. 

Here is a brief overview of the main steps followed: 
- Extracting movie titles: given an input text, output a list of possible movie titles that are mentionned in text. 
- Find movie associated with the title: returning a list of indices in the movies dataset corresponding to the movie titles extracted. 
- Extract sentiment from the input: using a dataset of common words associated with the corresponding sentiment. 
- Recommend movies: given a vector of the user's preferences and given ratings by other users, use collaborative filtering to return a list of movies with the highest recommendation score that the user hasn't seen, using collaborative filtering with cosine-similarity. 

## The course content

- Text Processing using Regular Expressions, Text Normalization and Edit Distance
- Langauge Modeling with N-grams
- Sentiment Classification using Naïve Bayes or Logistic Regression
- Information Retrieval 
- Vector Semantics and Embeddings 
- Neural Networks 
- Dialog Systems and Chatbots 
- Recommender Systems 

## The assignments 

Parallel to courses were assignments, which aimed to practice the theoretical concepts on real datasets. 

### Assignment 1 : SpamLord 
This assignment was about using regular expressions to extract phone numbers and email addresses from documents found on the web. 

### Assignment 2 : Triage 
The goal was to perform text classification on messages sent in the aftermath of natural disasters, using Multinomial Naive Bayes Classifier with add-1 smoothing, and Logistic Regression. Specifically, the aim was to determine whether a specific message was about aid. 

### Assigment 3 : Information Retrieval 
The goal was to build an inverted index to quickly retrieve documents that match queries and then make it better by using term-frequency inverse-document-frequency weighting and cosine similarity to compare queries to your data set. 

### Assignment 4 : Quizlet 
This homework waq about leveraging word embeddings to write a program that can answer various multiple choice and true/false quiz questions.

### Assignment 5 : Neural Networks 
Getting a taste of how NN work and why they are useful. 


 

 
