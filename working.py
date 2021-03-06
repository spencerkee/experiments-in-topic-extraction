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
import functions as f


def create_document_list(
        movie_id_to_movie,
        movie_id_to_rating,
        user_rating_history,
        threshold=5):
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
                if rated_movie[0] == movie_id or (
                        float(rated_movie[1]) < threshold):  # don't want to have cyclical associations
                    continue
                # doc_tags_list.append(rated_movie[0])
                title = movie_id_to_movie[rated_movie[0]].replace(" ", "_")
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
        print('Current # of topics: ' + str(num_topics))
        lm = models.LdaModel(
            corpus=corpus,
            num_topics=num_topics,
            id2word=dictionary)
        # lm_list.append(lm)
        cm = models.CoherenceModel(
            model=lm,
            texts=texts,
            dictionary=dictionary,
            coherence='c_v')
        print(str((cm.get_coherence())))
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


def replace_movie_id_with_name(string, movie_id_to_movie):
    string = string.split('"')
    string = [
        unicodedata.normalize(
            'NFKD',
            i).encode(
            'ascii',
            'ignore') for i in string]
    for ind, value in enumerate(string):
        if '.' not in value and len(value) > 0:
            string[ind] = movie_id_to_movie[value]
    return ''.join(string)

def center_user_ratings(user_ratings):
    for key in user_ratings:
        avg = sum([i[1] for i in user_ratings[key]])/len(user_ratings[key])
        for i in range(len(user_ratings[key])):
            user_ratings[key][i][1] -= avg
    return user_ratings

def main(name):
    # utility dictionaries
    movie_id_to_movie = f.create_movie_id_to_movie()  # movie id: movie title
    # user id: [(movie id,movie rating), etc]
    user_rating_history = f.create_user_rating_history()
    user_rating_history = center_user_ratings(user_rating_history)

    movie_id_to_rating = f.create_movie_id_to_rating(
        user_rating_history, movie_id_to_movie)  # movie id: [(user id, rating), etc]

    texts = [
        i for i in create_document_list(
            movie_id_to_movie,
            movie_id_to_rating,
            user_rating_history,
            threshold=0) if len(i) >= 0]

    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # running and saving models
    # ldamodel = models.ldamodel.LdaModel(corpus, num_topics=topic_num, id2word = dictionary, passes=10)
    # topics = ldamodel.show_topics(num_topics=topic_num, num_words=6)
    # ldamodel.save(fname=("may_results/" + name))

    # hdpmodel = models.HdpModel(corpus, id2word=dictionary)
    hdpmodel = models.hdpmodel.HdpModel.load(fname=("may_results/hdp-centered_threshold_0-movies_all_small-len_0"))
    # topics = hdpmodel.print_topics(num_topics=-1,num_words=6)
    # hdpmodel.save(fname_or_handle=("may_results/" + name))
    # for i in topics:
    #     print(i)

    ids = sorted(movie_id_to_rating.keys())
    data = []
    for i in range(len(texts)):
        vec = [0]*150
        for feature in hdpmodel[corpus[i]]:
            # try:
            vec[feature[0]] = feature[1]
            # except IndexError:
            #     print('feature',feature)
        row = [ids[i],movie_id_to_movie[ids[i]],vec]
        data.append(row)
        # print(hdpmodel[corpus[i]])
        # print(len(texts[i]))
        # if len(texts[i]) > 0:
        #     print(hdpmodel[corpus[i]])

    for i in data:
        print(i)


    # ldamodel = models.ldamodel.LdaModel.load(fname=("active_variables/threshold_5-movies_all_small-len_100-topics_20-passes_20"))
    # ldamodel.update(corpus, passes=10)
    # ldamodel.save(fname=("active_variables/" + name))
    # topics = ldamodel.print_topics(num_topics=topic_num, num_words=6)


    # don't think this works
    # filename = 'present/' + name
    # with open(filename,'wb') as f:
    #     for topic in topics:
    #         f.write(str(topic[0]) + ' ' + str(replace_movie_id_with_name(topic[1],movie_id_to_movie) + '\n'))




    # lm_list = evaluate_graph(dictionary, corpus, texts, 10, 101, 10)
    # lm_list_savename = 'current_results/' + name + '.p'
    # pickle.dump(lm_list,open(lm_list_savename, "wb"))

if __name__ == '__main__':
    start_time = time.time()
    topic_num = 20
    main(name='hdp-centered_threshold_0-movies_all_small-len_0')
    print("total time: {} seconds".format(str(time.time() - start_time)))

    # need to test this with actual text documents
