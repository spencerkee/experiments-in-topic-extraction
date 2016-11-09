from gensim import corpora, models
import gensim
import pickle
from process_files import create_user_rating_history, create_movie_id_to_movie
import unicodedata
import sys
import time

def create_movie_id_to_rating(user_rating_history,movie_id_to_movie):
    movie_id_to_rating = {} #movie id: [(user id, rating), etc]
    for user in user_rating_history:
        for rating in user_rating_history[user]:
            try:
                movie_id_to_rating[rating[0]].append((user,rating[1]))
            except KeyError:
                movie_id_to_rating[rating[0]] = [(user,rating[1])]
    return movie_id_to_rating

def create_document_list(movie_id_to_rating,user_rating_history):
    doc_set = []
    movies = sorted(movie_id_to_rating.keys()) #not movie_id_to_movie because some movies have no ratings?
    for movie in movies:#can be range(num_movies), for every movie
        doc_tags_list = []
        for user_rating in movie_id_to_rating[movie]:
            if float(user_rating[1]) >= 4.0:
                for rated_movie in user_rating_history[user_rating[0]]:
                    if rated_movie[0] == movie or (float(rated_movie[1]) >= 4): #don't want to have cyclical associations
                        continue
                    doc_tags_list.append(rated_movie[0])
        # doc_tags_list = sorted(doc_tags_list, key=int) #possibly remove, bad for preformance and only in for visualization purposes
        doc_set.append(doc_tags_list)
    # doc_set = []
    #for every movie:
    #for every user who rated the movie over 3:
    #for every movie that user has rated over 3:
    #add that movie to that movie's document (1) times
    return doc_set

def replace_movie_id_with_name(string,movie_id_to_movie):
    string = string.split('"')
    string = [unicodedata.normalize('NFKD', i).encode('ascii','ignore') for i in string]
    for ind,value in enumerate(string):
        if '.' not in value and len(value)>0:
            string[ind] = movie_id_to_movie[value]
    return ''.join(string)

def main(load=['texts','dictionary','corpus','ldamodel'],save=True,topic_num=20,passes_num=20):
    start_time = time.time()
    designation = str(topic_num) + "_" + str(passes_num)

    #1 second
    movie_id_to_movie = create_movie_id_to_movie() #movie id: movie title
    user_rating_history = create_user_rating_history() #user id: [(movie id,movie rating),etc]
    movie_id_to_rating = create_movie_id_to_rating(user_rating_history,movie_id_to_movie)  #movie id: [(user id, rating), etc]

    print("texts start: {} seconds".format(str(time.time()-start_time)))
    if 'texts' in load: #37 seconds
        texts = pickle.load(open( "texts"+designation+".p", "rb" ))
    else: #78 seconds
        texts = create_document_list(movie_id_to_rating,user_rating_history)
        if save: pickle.dump(texts, open( "texts"+designation+".p", "wb" ))

    print("dictionary start: {} seconds".format(str(time.time()-start_time)))
    if 'dictionary' in load: #2 seconds
        dictionary = pickle.load(open("dictionary"+designation+".p", "rb" ))
    else:
        # turn our tokenized documents into a id <-> term dictionary
        dictionary = corpora.Dictionary(texts)
        if save: pickle.dump(dictionary, open("dictionary"+designation+".p", "wb" ))

    print("corpus start: {} seconds".format(str(time.time()-start_time)))
    if 'corpus' in load: #172 seconds
        corpus = pickle.load(open( "corpus"+designation+".p", "rb" ))
    else: #511 seconds
        # convert tokenized documents into a document-term matrix
        corpus = [dictionary.doc2bow(text) for text in texts]
        if save: pickle.dump(corpus, open("corpus"+designation+".p", "wb" ))

    print("ldamodel start: {} seconds".format(str(time.time()-start_time)))
    lda_savename = "ldamodel" + designation + ".p"
    if 'ldamodel' in load:
        ldamodel = pickle.load(open(lda_savename, "rb" ))
    else:
        # generate LDA model
        ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=topic_num, id2word = dictionary, passes=20)
        if save: pickle.dump(ldamodel, open(lda_savename, "wb" ))

    topics = ldamodel.print_topics(num_topics=topic_num, num_words=6)

    filename = 'results/' + designation
    with open(filename,'wb') as f:
        for topic in topics:
            f.write(str(topic[0]) + ' ' + str(replace_movie_id_with_name(topic[1],movie_id_to_movie) + '\n'))

    print("total time: {} seconds".format(str(time.time()-start_time)))
if __name__ == '__main__':
    if len(sys.argv) != 4:
        sys.exit('usage [topic_num passes_num save]')
    if sys.argv[3] == 'True':
        save=True
    else:
        save=False
    main(load=[],save=save,topic_num=int(sys.argv[1]),passes_num=int(sys.argv[2]))
