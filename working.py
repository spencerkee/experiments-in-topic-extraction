import operator
from collections import namedtuple


from gensim import corpora, models, similarities

import matplotlib.pyplot as plt

from process_files import create_user_rating_history, create_movie_id_to_movie
import pickle
import time
import unicodedata

import os
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

def create_document_list(movie_id_to_movie, movie_id_to_rating,user_rating_history,threshold=5):
    """
    for every movie:
    for every user who rated the movie greater than or equal to threshold:
    for every movie that user has rated greater than or equal to threshold:
    add that movie to that movie's document (1) times
    """
    doc_set = []
    movies = sorted(movie_id_to_rating.keys())
    for movie_id in movies:
        doc_tags_list = []
        for user_rating in movie_id_to_rating[movie_id]:
            if float(user_rating[1]) < threshold:
                continue
            for rated_movie in user_rating_history[user_rating[0]]:
                if rated_movie[0] == movie_id or (float(rated_movie[1]) < threshold): #don't want to have cyclical associations
                    continue
                # doc_tags_list.append(rated_movie[0])
                title = movie_id_to_movie[rated_movie[0]].replace(" ","_")
                doc_tags_list.append(title)
        doc_set.append(doc_tags_list)
    return doc_set

def range_list(start, limit, inc=1):
    return_list = []
    val = start
    while val < limit:
        return_list.append(val)
        val += inc
    return return_list

def evaluate_graph(dictionary, corpus, texts, start_val, limit, step=1):
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
    num_topics = start_val
    while num_topics < limit:
        print ('Current # of topics: ' + str(num_topics))
        lm = models.LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary)
        # lm_list.append(lm)
        cm = models.CoherenceModel(model=lm, texts=texts, dictionary=dictionary, coherence='c_v')
        print (str((cm.get_coherence())))
        c_v.append(cm.get_coherence())
        num_topics += step

    # Show graph
    x = range_list(start_val, limit, step)
    plt.plot(x, c_v)
    plt.xlabel("num_topics")
    plt.ylabel("Coherence score")
    plt.legend(("c_v"), loc='best')
    plt.show()

    return lm_list

def replace_movie_id_with_name(string,movie_id_to_movie):
    string = string.split('"')
    string = [unicodedata.normalize('NFKD', i).encode('ascii','ignore') for i in string]
    for ind,value in enumerate(string):
        if '.' not in value and len(value)>0:
            string[ind] = movie_id_to_movie[value]
    return ''.join(string)

def main(name):
    #utility dictionaries
    movie_id_to_movie = create_movie_id_to_movie() #movie id: movie title
    user_rating_history = create_user_rating_history() #user id: [(movie id,movie rating), etc]
    movie_id_to_rating = create_movie_id_to_rating(user_rating_history,movie_id_to_movie)  #movie id: [(user id, rating), etc]

    #active variables
    texts = [i for i in create_document_list(movie_id_to_movie, movie_id_to_rating,user_rating_history, threshold=4) if len(i) >= 0]
    # word_counter = {}
    # for word in texts[49]:
    #     if word in word_counter:
    #         word_counter[word] += 1
    #     else:
    #         word_counter[word] = 1
    # popular_words = sorted(word_counter, key = word_counter.get, reverse = True)
    # top = popular_words[:10]
    # print(top)

    print (texts[0])

    # dictionary = corpora.Dictionary(texts)
    # corpus = [dictionary.doc2bow(text) for text in texts]
    # # print (corpus[0])
    # tfidf = models.TfidfModel(corpus)
    # data = []
    # for i in tfidf[corpus[0]]:
    #     data.append([i[1],dictionary[i[0]]])
    # data.sort(reverse=True)
    # int(len(data)/4)
    # print (data[:int(len(data)/4)])
    # data.sort()
    # print(data[:int(len(data)/4)])


    # saving active variables
    # variables = [texts,dictionary,corpus]
    # pickle.dump(variables,open('active_variables/' + name + '-variables.p', "wb"))
    # texts, dictionary, corpus = pickle.load(open('active_variables/threshold_5-movies_all_small-len_100-variables.p', 'rb'))

    # running and saving models
    # ldamodel = models.ldamodel.LdaModel(corpus, num_topics=topic_num, id2word = dictionary, passes=10)
    # topics = ldamodel.show_topics(num_topics=topic_num, num_words=6)
    # ldamodel.save(fname=("active_variables/" + name))

    # ldamodel = models.ldamodel.LdaModel.load(fname=("active_variables/threshold_5-movies_all_small-len_100-topics_20-passes_20"))
    # ldamodel.update(corpus, passes=10)
    # ldamodel.save(fname=("active_variables/" + name))
    # topics = ldamodel.print_topics(num_topics=topic_num, num_words=6)
    # filename = 'current_results/' + name
    # with open(filename,'wb') as f:
    #     for topic in topics:
    #         f.write(str(topic[0]) + ' ' + str(replace_movie_id_with_name(topic[1],movie_id_to_movie) + '\n'))

    # lm_list = evaluate_graph(dictionary, corpus, texts, 10, 101, 10)
    # lm_list_savename = 'current_results/' + name + '.p'
    # pickle.dump(lm_list,open(lm_list_savename, "wb"))

if __name__ == '__main__':
    start_time = time.time()
    topic_num = 20
    main(name='threshold_5-movies_all_small-len_100-topics_20-passes_30')
    print("total time: {} seconds".format(str(time.time()-start_time)))

    #need to test this with actual text documents
