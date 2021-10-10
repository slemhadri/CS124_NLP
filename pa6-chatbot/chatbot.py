# PA6, CS124, Stanford, Winter 2019
# v.1.0.3
# Original Python code by Ignacio Cases (@cases)

import util

import numpy as np

# for regular expressions
import re

# for reading files
from io import open
import os

#stemming words
from porter_stemmer import PorterStemmer

from collections import defaultdict

# noinspection PyMethodMayBeStatic
class Chatbot:
    """Simple class to implement the chatbot for PA 6."""

    def __init__(self, creative=False):
        # The chatbot's default name is `moviebot`.
        # TODO: Give your chatbot a new name.
        self.name = "Selma's MovieBot"

        self.creative = creative
        # This matrix has the following shape: num_movies x num_users
        # The values stored in each row i and column j is the rating for
        # movie i by user j
        self.titles, ratings = util.load_ratings('data/ratings.txt')
        self.sentiment = util.load_sentiment_dictionary('data/sentiment.txt')
        self.p = PorterStemmer()

        ########################################################################
        # TODO: Binarize the movie ratings matrix.                             #
        ########################################################################

        # Binarize the movie ratings before storing the binarized matrix.
        self.ratings = self.binarize(ratings, 2.5) # check if correct
        self.num_movies = np.shape(self.ratings)[0]
        self.users_ratings = np.zeros(self.num_movies)
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    ############################################################################
    # 1. WARM UP REPL                                                          #
    ############################################################################

    def greeting(self):
        """Return a message that the chatbot uses to greet the user."""
        ########################################################################
        # TODO: Write a short greeting message                                 #
        ########################################################################

        greeting_message = "Hello. I'm Selma's MovieBot. I'll recommend a movie for you to watch. Tell me about a movie you watched."

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        return greeting_message

    def goodbye(self):
        """
        Return a message that the chatbot uses to bid farewell to the user.
        """
        ########################################################################
        # TODO: Write a short farewell message                                 #
        ########################################################################

        goodbye_message = "I hope this was useful. Enjoy your movie and see you soon!"

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return goodbye_message

    ############################################################################
    # 2. Modules 2 and 3: extraction and transformation                        #
    ############################################################################

    def process(self, line):
        """Process a line of input from the REPL and generate a response.

        This is the method that is called by the REPL loop directly with user
        input.

        You should delegate most of the work of processing the user's input to
        the helper functions you write later in this class.

        Takes the input string from the REPL and call delegated functions that
          1) extract the relevant information, and
          2) transform the information into a response to the user.

        Example:
          resp = chatbot.process('I loved "The Notebook" so much!!')
          print(resp) // prints 'So you loved "The Notebook", huh?'

        :param line: a user-supplied line of text
        :returns: a string containing the chatbot's response to the user input
        """
        ########################################################################
        # TODO: Implement the extraction and transformation in this method,    #
        # possibly calling other functions. Although your code is not graded   #
        # directly based on how modular it is, we highly recommended writing   #
        # code in a modular fashion to make it easier to improve and debug.    #
        ########################################################################
        if self.creative:
            response = "I processed {} in creative mode!!".format(line)

        else:
            response = "I processed {} in starter mode!!".format(line)

        movie_sentiments = self.extract_sentiment_for_movies(line)

        if movie_sentiments == []:
            return "I'm sorry, I have never heard of this movie. Can you tell me about another one?"

        list_tuple_mov_sent = []

        for movie_sentiment in movie_sentiments:
            movie_IDS = self.find_movies_by_title(movie_sentiment[0])

            while len(movie_IDS) > 1:
                clarification = input("Did you mean: " + self.recommend_for_user(movie_IDS) + '?\n> ')
                movie_IDS = self.disambiguate(clarification, movie_IDS)

            for movie_ID in movie_IDS:
                list_tuple_mov_sent.append((movie_ID, movie_sentiment[1]))

        if list_tuple_mov_sent == []:
            return "I'm sorry, I have never heard of this movie. Can you tell me about another one?"

        dict_mov_sent = self.create_dict_user(list_tuple_mov_sent)
        response += self.get_acknowledgment_movies(dict_mov_sent)

        if sum(self.users_ratings != 0) < 5:
            response = response + "\nPlease tell me about another movie you liked/disliked."
        else:
            print("\nI have anough information to recommend a movie! Loading...")
            recommendations = self.recommend(self.users_ratings, self.ratings)
            response = response + "\nYou might like these movies: " + self.recommend_for_user(recommendations)

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return response

    def recommend_for_user(self, recommendations):
        movie_titles_recom = [self.get_movie_title_for_user(movie_ID) for movie_ID in recommendations]
        movies = ''
        for movie_title in movie_titles_recom:
            movies += movie_title + ', '
        return movies

    def get_acknowledgment_movies(self, dict_mov_sent):
        acknowledgments = ''
        for movie_title in dict_mov_sent:
            acknowledgments += '\n' + self.acknowledge(movie_title, dict_mov_sent[movie_title])
        return acknowledgments

    def create_dict_user(self, list_tuple_mov_sent):
        dict_mov_sent = {}
        """
        for movie_sentiment in movie_sentiments:
            movie_IDS = self.find_movies_by_title(movie_sentiment[0])
            for movie in movie_IDS:
                self.users_ratings[movie] = movie_sentiment[1]
                movie_title = self.get_movie_title_for_user(movie)
                dict_mov_sent[movie_title] = movie_sentiment[1]
        """
        for tuple_movie in list_tuple_mov_sent:
            self.users_ratings[tuple_movie[0]] = tuple_movie[1]
            movie_title = self.get_movie_title_for_user(tuple_movie[0])
            dict_mov_sent[movie_title] = tuple_movie[1]
        return dict_mov_sent

    @staticmethod
    def preprocess(text):
        """Do any general-purpose pre-processing before extracting information
        from a line of text.

        Given an input line of text, this method should do any general
        pre-processing and return the pre-processed string. The outputs of this
        method will be used as inputs (instead of the original raw text) for the
        extract_titles, extract_sentiment, and extract_sentiment_for_movies
        methods.

        Note that this method is intentially made static, as you shouldn't need
        to use any attributes of Chatbot in this method.

        :param text: a user-supplied line of text
        :returns: the same text, pre-processed
        """
        ########################################################################
        # TODO: Preprocess the text into a desired format.                     #
        # NOTE: This method is completely OPTIONAL. If it is not helpful to    #
        # your implementation to do any generic preprocessing, feel free to    #
        # leave this method unmodified.                                        #
        ########################################################################

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

        return text

    def acknowledge(self, movie_title, sentiment):

        if sentiment == 1:
            acknowledgment = "You liked " + movie_title + '.'
        elif sentiment == -1:
            acknowledgment = "You didn't like " + movie_title + '.'
        else:
            acknowledgment = "I didn't understand if you liked " + movie_title + "or not. Please tell me more about " + movie_title + '.'

        return acknowledgment

    def get_movie_title_for_user(selfself, movie_ID):
        pattern = str(movie_ID) + '%(.+)%'
        with open(os.path.join('data', 'movies.txt'), 'r', encoding = 'ISO-8859-1') as file:
            all_txt = file.read()
            matches = re.findall(pattern, all_txt)

        return matches[0]

    def extract_titles(self, preprocessed_input):
        """Extract potential movie titles from a line of pre-processed text.

        Given an input text which has been pre-processed with preprocess(),
        this method should return a list of movie titles that are potentially
        in the text.

        - If there are no movie titles in the text, return an empty list.
        - If there is exactly one movie title in the text, return a list
        containing just that one movie title.
        - If there are multiple movie titles in the text, return a list
        of all movie titles you've extracted from the text.

        Example:
          potential_titles = chatbot.extract_titles(chatbot.preprocess(
                                            'I liked "The Notebook" a lot.'))
          print(potential_titles) // prints ["The Notebook"]

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: list of movie titles that are potentially in the text
        """
        pattern = '[a-zA-Z\s\.,\?]*(?:"([^".]*)"(?:[a-zA-Z\s\.,\?]"(.*)")*)?'
        # list of strings that match the capture groups
        match_group = re.findall(pattern, preprocessed_input)
        list_matches = [match_group[i][0] for i in range(len(match_group) - 1) if match_group[i][0] != '']

        return list_matches

    def find_movies_by_title(self, title):
        """ Given a movie title, return a list of indices of matching movies.

        - If no movies are found that match the given title, return an empty
        list.
        - If multiple movies are found that match the given title, return a list
        containing all of the indices of these matching movies.
        - If exactly one movie is found that matches the given title, return a
        list
        that contains the index of that matching movie.

        Example:
          ids = chatbot.find_movies_by_title('Titanic')
          print(ids) // prints [1359, 2716]

        :param title: a string containing a movie title
        :returns: a list of indices of matching movies
        """
        title = title.lower()
        title.strip()
        title_list = title.split(' ')
        list_beginnings = ['the', 'an', 'a', 'les', 'le', 'la', 'un', 'une', 'el']
        year = ''

        if title_list[-1][0] == '(':
            year = title_list[-1].strip('(')
            year = year.strip(')')
            title_list.pop(-1)
            title = ''
            for i in range(len(title_list)):
                title += title_list[i]
                if i != len(title_list) - 1:
                    title += ' '

        if title_list[0] in list_beginnings:
            title = ''
            for i in range(1, len(title_list)):
                title += title_list[i]
                if i != len(title_list) - 1:
                    title += ' '

            title += ', ' + title_list[0]

        if year != '':
            pattern = '(\d+)%' + title + '(?:: .+ | (?:\([^\(\)]+\) )?)?' + '\(' + year + '\)'
        else:
            pattern = '(\d+)%' + title + '(?:: .+| (?:\([^\(\)]+\) )?\((?:\d)+\))'
        with open(os.path.join('data', 'movies.txt'), 'r', encoding = 'ISO-8859-1') as file:
            all_txt = (file.read()).lower()
            matches = re.findall(pattern, all_txt)

        ids = [int(matches[i]) for i in range(len(matches))]

        return ids

    def extract_sentiment(self, preprocessed_input):
        """Extract a sentiment rating from a line of pre-processed text.

        You should return -1 if the sentiment of the text is negative, 0 if the
        sentiment of the text is neutral (no sentiment detected), or +1 if the
        sentiment of the text is positive.

        As an optional creative extension, return -2 if the sentiment of the
        text is super negative and +2 if the sentiment of the text is super
        positive.

        Example:
          sentiment = chatbot.extract_sentiment(chatbot.preprocess(
                                                    'I liked "The Titanic"'))
          print(sentiment) // prints 1

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: a numerical value for the sentiment of the text
        """

        """
        create a dictionnary from sentiment.txt where the keys are the stemmed 
        words and the value is 1 if pos and -1 if neg. 
        """

        # BUG to resolve: the default dict associates to unknown movies a neutral sentiment
        # automatically

        dict_sentiment = defaultdict(int)

        with open(os.path.join('data', 'sentiment.txt'), 'r', encoding='ISO-8859-1') as file:
            lines = file.readlines()
            for line in lines:
                word_stm_list = line.split(',')
                word_input = word_stm_list[0].strip()
                sentiment = 2 * (word_stm_list[1].strip() == "pos") - 1
                dict_sentiment[self.p.stem(word_input)] = sentiment


        negations = {"no", "not", "rather", "couldn't", "wasn't", "didn't", "wouldn't", "shouldn't", "weren't", "don't",
                     "doesn't", "haven't", "hasn't", "won't", "wont", "hadn't", "never", "none", "nobody", "nothing",
                     "neither", "nor", "nowhere", "isn't", "can't", "cannot", "mustn't", "mightn't", "shan't", "without",
                     "needn't"}

        """     
        idea: if one word in negations (set of negation words) occurs, change sentiment associated with 
        the following words up until the end of the sentence. 
        """

        movie_sentence = self.extract_sentence(preprocessed_input)
        # turn the input into a list of words stripped from all delimiters.
        delimiters = '; ?|, ?|\s|!+ ?|\?|\. ?'
        list_words = re.split(delimiters, movie_sentence[1])

        sentiment = 0

        neg_const = 1
        for word in list_words:
            word = word.lower()
            if word not in negations:
                word = self.p.stem(word)
                sentiment += neg_const*dict_sentiment[word]

            else:
                neg_const *= -1

        return sentiment

    def extract_sentence(self, preprocessed_input):
        movie_title = self.extract_titles(preprocessed_input)
        sentence = ''
        if movie_title != []:
            start_index = preprocessed_input.find(movie_title[0]) - 1
            for i in range(start_index):
                sentence += preprocessed_input[i]
            if len(preprocessed_input) > start_index + len(movie_title) + 2:
                for i in range(start_index + len(movie_title[0]) + 2, len(preprocessed_input)):
                    sentence += preprocessed_input[i]
        movie_sentence = (movie_title, sentence)
        return movie_sentence

    def extract_sentiment_for_movies(self, preprocessed_input):
        """Creative Feature: Extracts the sentiments from a line of
        pre-processed text that may contain multiple movies. Note that the
        sentiments toward the movies may be different.

        You should use the same sentiment values as extract_sentiment, described

        above.
        Hint: feel free to call previously defined functions to implement this.

        Example:
          sentiments = chatbot.extract_sentiment_for_text(
                           chatbot.preprocess(
                           'I liked both "Titanic (1997)" and "Ex Machina".'))
          print(sentiments) // prints [("Titanic (1997)", 1), ("Ex Machina", 1)]

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: a list of tuples, where the first item in the tuple is a movie
        title, and the second is the sentiment in the text toward that movie
        """

        # For now. Accepts one movie per sentence in creative mode.

        sentiments = []
        input_sentences = preprocessed_input.split('.')
        for input_sentence in input_sentences:
            movie_title = self.extract_titles(input_sentence)[0]
            movie_IDS = self.find_movies_by_title(movie_title)
            if movie_IDS != []:
                sentiment = self.extract_sentiment(input_sentence)
                sentiments.append((movie_title, sentiment))

        return sentiments

    def find_movies_closest_to_title(self, title, max_distance=3):
        """Creative Feature: Given a potentially misspelled movie title,
        return a list of the movies in the dataset whose titles have the least
        edit distance from the provided title, and with edit distance at most
        max_distance.

        - If no movies have titles within max_distance of the provided title,
        return an empty list.
        - Otherwise, if there's a movie closer in edit distance to the given
        title than all other movies, return a 1-element list containing its
        index.
        - If there is a tie for closest movie, return a list with the indices
        of all movies tying for minimum edit distance to the given movie.

        Example:
          # should return [1656]
          chatbot.find_movies_closest_to_title("Sleeping Beaty")

        :param title: a potentially misspelled title
        :param max_distance: the maximum edit distance to search for
        :returns: a list of movie indices with titles closest to the given title
        and within edit distance max_distance
        """

        pass

    def disambiguate(self, clarification, candidates):
        """Creative Feature: Given a list of movies that the user could be
        talking about (represented as indices), and a string given by the user
        as clarification (eg. in response to your bot saying "Which movie did
        you mean: Titanic (1953) or Titanic (1997)?"), use the clarification to
        narrow down the list and return a smaller list of candidates (hopefully
        just 1!)

        - If the clarification uniquely identifies one of the movies, this
        should return a 1-element list with the index of that movie.
        - If it's unclear which movie the user means by the clarification, it
        should return a list with the indices it could be referring to (to
        continue the disambiguation dialogue).

        Example:
          chatbot.disambiguate("1997", [1359, 2716]) should return [1359]

        :param clarification: user input intended to disambiguate between the
        given movies
        :param candidates: a list of movie indices
        :returns: a list of indices corresponding to the movies identified by
        the clarification
        """
        list_movie_titles = [self.get_movie_title_for_user(x).lower() for x in candidates]
        clarification = clarification.lower()
        result_IDs = []
        for i in range(len(list_movie_titles)):
            if clarification in list_movie_titles[i]:
                result_IDs.append(candidates[i])
        return result_IDs

    ############################################################################
    # 3. Movie Recommendation helper functions                                 #
    ############################################################################

    @staticmethod
    def binarize(ratings, threshold=2.5):
        """Return a binarized version of the given matrix.

        To binarize a matrix, replace all entries above the threshold with 1.
        and replace all entries at or below the threshold with a -1.

        Entries whose values are 0 represent null values and should remain at 0.

        Note that this method is intentionally made static, as you shouldn't use
        any attributes of Chatbot like self.ratings in this method.

        :param ratings: a (num_movies x num_users) matrix of user ratings, from
         0.5 to 5.0
        :param threshold: Numerical rating above which ratings are considered
        positive

        :returns: a binarized version of the movie-rating matrix
        """
        ########################################################################
        # TODO: Binarize the supplied ratings matrix.                          #
        #                                                                      #
        # WARNING: Do not use self.ratings directly in this function.          #
        ########################################################################

        # The starter code returns a new matrix shaped like ratings but full of
        # zeros.

        #is_zero = ratings == 0
        binarized_ratings = np.empty_like(ratings)
        binarized_ratings[ratings <= threshold] = -1
        binarized_ratings[ratings > threshold] = 1
        #ratings[is_zero] = 0
        binarized_ratings[ratings == 0] = 0

        #binarized_ratings = 2 * (ratings > threshold) - 1

        # DEBUG THIS

        ########################################################################
        #                        END OF YOUR CODE                              #
        ########################################################################
        return binarized_ratings

    def similarity(self, u, v):
        """Calculate the cosine similarity between two vectors.

        You may assume that the two arguments have the same shape.

        :param u: one vector, as a 1D numpy array
        :param v: another vector, as a 1D numpy array

        :returns: the cosine similarity between the two vectors
        """
        ########################################################################
        # TODO: Compute cosine similarity between the two vectors.             #
        ########################################################################
        norm_u_v = np.sqrt(np.dot(u, u) * np.dot(v, v))
        if norm_u_v == 0:
            return 0
        return np.dot(u, v) / norm_u_v
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################


    def recommend(self, user_ratings, ratings_matrix, k=10, creative=False):
        """Generate a list of indices of movies to recommend using collaborative
         filtering.

        You should return a collection of `k` indices of movies recommendations.

        As a precondition, user_ratings and ratings_matrix are both binarized.

        Remember to exclude movies the user has already rated!

        Please do not use self.ratings directly in this method.

        :param user_ratings: a binarized 1D numpy array of the user's movie
            ratings
        :param ratings_matrix: a binarized 2D numpy matrix of all ratings, where
          `ratings_matrix[i, j]` is the rating for movie i by user j
        :param k: the number of recommendations to generate
        :param creative: whether the chatbot is in creative mode

        :returns: a list of k movie indices corresponding to movies in
        ratings_matrix, in descending order of recommendation.
        """

        ########################################################################
        # TODO: Implement a recommendation function that takes a vector        #
        # user_ratings and matrix ratings_matrix and outputs a list of movies  #
        # recommended by the chatbot.                                          #
        #                                                                      #
        # WARNING: Do not use the self.ratings matrix directly in this         #
        # function.                                                            #
        #                                                                      #
        # For starter mode, you should use item-item collaborative filtering   #
        # with cosine similarity, no mean-centering, and no normalization of   #
        # scores.                                                              #
        ########################################################################

        """
        Correction: initially, user_ratings is 1-D array of size num_movies, where
        the value is 0 if the user hasn't rated the movie, 1 if he/she liked it, 
        -1 if he/she didn't like it. 
        """
        dict_sentiments = {}

        num_movies = np.shape(ratings_matrix)[0]

        for movie_ID in range(num_movies):
            dict_sentiments[movie_ID] = user_ratings[movie_ID]

        """
        For each movie_ID in the dataset (i.e. each row of the matrix, except the ones in 
        dict_sentiments keys, calculate the expected rating that the user would have gave 
        it, using cosine similarity between the movie and the movies rated by the user, 
        and the rating the user gave to each movie. 
        """

        # find the movies that the user rated.
        user_rated_movies_ID = set()

        for movie_ID in dict_sentiments:
            if dict_sentiments[movie_ID] != 0:
                user_rated_movies_ID.add(movie_ID)

        # get the recommendations
        expected_ratings = {}

        for i in range(num_movies):
            similarities = np.array([])
            ratings = np.array([])
            if i not in user_rated_movies_ID:
                for rated_movie_ID in user_rated_movies_ID:
                    cos_sim = self.similarity(ratings_matrix[i], ratings_matrix[rated_movie_ID])
                    similarities = np.append(similarities, cos_sim)
                    ratings = np.append(ratings, dict_sentiments[rated_movie_ID])
            expected_rating = np.dot(similarities, ratings)
            expected_ratings[i] = expected_rating


        # Populate this list with k movie indices to recommend to the user.
        recommendations = []

        for i in range(k):
            max_movie = list(expected_ratings.keys())[list(expected_ratings.values()).index(max(expected_ratings.values()))]
            recommendations.append(max_movie)
            del expected_ratings[max_movie]

        ########################################################################
        #                        END OF YOUR CODE                              #
        ########################################################################
        return recommendations

    ############################################################################
    # 4. Debug info                                                            #
    ############################################################################

    def debug(self, line):
        """
        Return debug information as a string for the line string from the REPL

        NOTE: Pass the debug information that you may think is important for
        your evaluators.
        """
        debug_info = 'debug info'
        return debug_info

    ############################################################################
    # 5. Write a description for your chatbot here!                            #
    ############################################################################
    def intro(self):
        """Return a string to use as your chatbot's description for the user.

        Consider adding to this description any information about what your
        chatbot can do and how the user can interact with it.
        """
        return """
        Your task is to implement the chatbot as detailed in the PA6
        instructions.
        Remember: in the starter mode, movie names will come in quotation marks
        and expressions of sentiment will be simple!
        TODO: Write here the description for your own chatbot!
        """


if __name__ == '__main__':
    print('To run your chatbot in an interactive loop from the command line, '
          'run:')
    print('    python3 repl.py')
