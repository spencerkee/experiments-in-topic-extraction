import pickle

def create_user_rating_history():
    user_rating_history = {}
    with open('ratings.csv','r') as f:
        for line in f:
            if 'userId,movieId,rating,timestamp' in line:
                continue
            user_id, movie_id, rating= line.strip('\n').split(',')[:-1]
            try:
                user_rating_history[user_id].append((movie_id,rating))
            except KeyError:
                user_rating_history[user_id] = [(movie_id,rating)]
    # pickle.dump(user_rating_history, open( "user_rating_history.p", "wb" ))
    return user_rating_history

def create_movie_id_to_movie():
    movie_id_to_movie = {}
    with open('movies.csv','r') as f:
        for line in f:
            if 'movieId,title,genres' in line:
                continue
            line_data=line.strip('\n').split(',')
            if len(line_data) <= 3:
                movie_id, title = line_data[:2]
                movie_id_to_movie[movie_id] = title
            else:
                new_line_data = [line_data[0],''.join(line_data[1:-1])]
                movie_id, title = new_line_data
                movie_id_to_movie[movie_id] = title
    # pickle.dump(movie_id_to_movie, open( "movie_id_to_movie.p", "wb" ))
    return movie_id_to_movie