from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
import pattern
from gensim import corpora, models
import gensim
import pickle
from process_files import create_user_rating_history, create_movie_id_to_movie

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
    # for movie in sorted(movie_id_to_movie.keys()):#can be range(num_movies)
    #     doc_string = ''
    #     for 

    return doc_set

# texts = create_document_list()
# pickle.dump(texts, open( "texts.p", "wb" ))

# tokenizer = RegexpTokenizer(r'\w+')

# # create English stop words list
# en_stop = get_stop_words('en')

# # Create p_stemmer of class PorterStemmer
# p_stemmer = PorterStemmer()

# # create sample documents
# doc_a = "Brocolli is good to eat. My brother likes to eat good brocolli, but not my mother."
# doc_b = "My mother spends a lot of time driving my brother around to baseball practice."
# doc_c = "Some health experts suggest that driving may cause increased tension and blood pressure."
# doc_d = "I often feel pressure to perform well at school, but my mother never seems to drive my brother to do better."
# doc_e = "Health professionals say that brocolli is good for your health."

# # compile sample documents into a list
# doc_set = [doc_a, doc_b, doc_c, doc_d, doc_e]

# # list for tokenized documents in loop
# texts = []

# # loop through document list
# for i in doc_set:

#     # clean and tokenize document string
#     raw = i.lower()
#     tokens = tokenizer.tokenize(raw)
#     print("tokens: {}".format(tokens))

#     # remove stop words from tokens
#     stopped_tokens = [i for i in tokens if not i in en_stop]
#     print("stopped_tokens: {}".format(stopped_tokens))

#     # stem tokens
#     stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
#     print("stemmed_tokens: {}".format(stemmed_tokens))

#     # add tokens to list
#     texts.append(stemmed_tokens)
#     break

texts = pickle.load(open( "texts.p", "rb" ))

# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(texts)
pickle.dump(ldamodel, open( "dictionary.p", "wb" ))

# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in texts]
pickle.dump(ldamodel, open( "corpus.p", "wb" ))

# generate LDA model
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=20, id2word = dictionary, passes=20)
pickle.dump(ldamodel, open( "ldamodel.p", "wb" ))

for i in ldamodel.print_topics(num_topics=20, num_words=8):
    print (i)

#num_topics=20, passes=20