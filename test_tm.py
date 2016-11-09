from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
import pattern
from gensim import corpora, models
import gensim
import pickle
from process_files import create_user_rating_history, create_movie_id_to_movie
import sys

def create_document_list():
    movie_id_to_movie = create_movie_id_to_movie()#movie id: movie title
    user_rating_history = create_user_rating_history() #user id: [(movie id,movie rating),etc]
    movie_id_to_rating = {} #movie id: [(user id, rating), etc]
    for user in user_rating_history:
        # print ('user', user)
        # print ('users', user_rating_history[user])
        for rating in user_rating_history[user]:
            # print ('rating', rating)
            try:
                movie_id_to_rating[rating[0]].append((user,rating[1]))
            except KeyError:
                movie_id_to_rating[rating[0]] = [(user,rating[1])]

    doc_set = []
    movies = sorted(movie_id_to_rating.keys()) #not movie_id_to_movie because some movies have no ratings?
    for movie in movies:#can be range(num_movies), for every movie
        doc_tags_list = []
        # print (movie_id_to_rating[movie])
        for user_rating in movie_id_to_rating[movie]:
            if float(user_rating[1]) > 3.0:
                for rated_movie in user_rating_history[user_rating[0]]:
                    if rated_movie[0] == movie or (float(rated_movie[1]) > 3): #don't want to have cyclical associations
                        continue
                    doc_tags_list.append(rated_movie[0])
        doc_tags_list = sorted(doc_tags_list, key=int) #possibly remove, bad for preformance and only in for visualization purposes
        # print ('duplicates', len(doc_tags_list) != len(set(doc_tags_list)))
        # doc_set.append(' '.join(doc_tags_list))
        doc_set.append(doc_tags_list)
    # doc_set = []
    # #for every movie:
    # #for every user who rated the movie over 3:
    # #for every movie that user has rated over 3:
    # #add that movie to the document (1) times
    return doc_set

def main(load=['texts','dictionary','corpus','ldamodel'],save=True):
    # if 'texts' in load: #37 seconds
    #     texts = pickle.load(open( "texts.p", "rb" ))
    # else: #78 seconds
    #     texts = create_document_list()
    #     if save: pickle.dump(texts, open( "texts.p", "wb" ))

    # if 'dictionary' in load: #2 seconds
    #     dictionary = pickle.load(open( "dictionary.p", "rb" ))
    # else:
    #     # turn our tokenized documents into a id <-> term dictionary
    #     dictionary = corpora.Dictionary(texts)
    #     if save: pickle.dump(dictionary, open( "dictionary.p", "wb" ))

    # if 'corpus' in load:
    #     corpus = pickle.load(open( "corpus.p", "rb" ))
    # else: #511 seconds
    #     # convert tokenized documents into a document-term matrix
    #     corpus = [dictionary.doc2bow(text) for text in texts]
    #     if save: pickle.dump(corpus, open( "corpus.p", "wb" ))

    if 'ldamodel' in load:
        ldamodel = pickle.load(open( "ldamodel.p", "rb" ))
    else:
        # generate LDA model
        ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=20, id2word = dictionary, passes=20)
        if save: pickle.dump(ldamodel, open( "ldamodel.p", "wb" ))

    for i in ldamodel.print_topics(num_topics=20, num_words=8):
        print (i)

#num_topics=20, passes=20

if __name__ == '__main__':
    main(load=['texts','corpus', 'dictionary','ldamodel'],save=True)