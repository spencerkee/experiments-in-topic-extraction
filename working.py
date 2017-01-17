import operator
from collections import namedtuple

from gensim.models import LdaModel
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary

import matplotlib.pyplot as plt

from process_files import create_user_rating_history, create_movie_id_to_movie
import pickle
import time
import unicodedata

# %matplotlib inline

def create_movie_id_to_rating(user_rating_history,movie_id_to_movie):
    movie_id_to_rating = {} #movie id: [(user id, rating), etc]
    for user in user_rating_history:
        for rating in user_rating_history[user]:
            try:
                movie_id_to_rating[rating[0]].append((user,rating[1]))
            except KeyError:
                movie_id_to_rating[rating[0]] = [(user,rating[1])]
    return movie_id_to_rating

def create_document_list(movie_id_to_rating,user_rating_history,threshold=5):
    """
    for every movie:
    for every user who rated the movie greater than or equal to threshold:
    for every movie that user has rated greater than or equal to threshold:
    add that movie to that movie's document (1) times
    """
    doc_set = []
    movies = sorted(movie_id_to_rating.keys())
    for movie in movies:
        doc_tags_list = []
        for user_rating in movie_id_to_rating[movie]:
            if float(user_rating[1]) >= threshold:
                for rated_movie in user_rating_history[user_rating[0]]:
                    if rated_movie[0] == movie or (float(rated_movie[1]) >= threshold): #don't want to have cyclical associations
                        continue
                    doc_tags_list.append(rated_movie[0])
        doc_set.append(doc_tags_list)
    return doc_set

def evaluate_graph(dictionary, corpus, texts, limit):
    """
    Function to display num_topics - LDA graph using c_v coherence
    
    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    limit : topic limit
    
    Returns:
    -------
    lm_list : List of LDA topic models
    """
    c_v = []
    lm_list = []
    for num_topics in range(1, limit):
        lm = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary)
        lm_list.append(lm)
        cm = CoherenceModel(model=lm, texts=texts, dictionary=dictionary, coherence='c_v')
        c_v.append(cm.get_coherence())
        
    # Show graph
    x = range(1, limit)
    plt.plot(x, c_v)
    plt.xlabel("num_topics")
    plt.ylabel("Coherence score")
    plt.legend(("c_v"), loc='best')
    plt.show()
    
    return lm_list

def main(name):
    start_time = time.time()
    movie_id_to_movie = create_movie_id_to_movie() #movie id: movie title
    user_rating_history = create_user_rating_history() #user id: [(movie id,movie rating),etc]
    movie_id_to_rating = create_movie_id_to_rating(user_rating_history,movie_id_to_movie)  #movie id: [(user id, rating), etc]

    texts = [i for i in create_document_list(movie_id_to_rating,user_rating_history) if len(i) >= 100]
    dictionary = Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=topic_num, id2word = dictionary, passes=10)
    # topics = ldamodel.print_topics(num_topics=topic_num, num_words=6)

    lm_list = evaluate_graph(dictionary, corpus, texts, 3)

if __name__ == '__main__':
    main(name='test')