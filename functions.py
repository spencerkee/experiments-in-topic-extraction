def create_user_rating_history(filename='ratings.csv'):
    """
    Returns:
        user_rating_history ( dict - {string:[[string,float]]} ) -
        {user_id:[(movie_id,rating)]}
    """
    user_rating_history = {}
    with open('ratings.csv','r') as f:
        for line in f:
            if 'userId,movieId,rating,timestamp' in line:
                continue
            user_id, movie_id, rating= line.strip('\n').split(',')[:-1]
            try:
                user_rating_history[user_id].append([movie_id,float(rating)])
            except KeyError:
                user_rating_history[user_id] = [[movie_id,float(rating)]]
    return user_rating_history

def create_movie_id_to_movie(filename='movies.csv'):
    """
    Creates a dictionary where the keys are movie ids and the values are movies
    Both are strings
    """
    movie_id_to_movie = {}
    with open(filename,'r') as f:
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
    return movie_id_to_movie

def create_movie_id_to_rating(user_rating_history,movie_id_to_movie):
    """
    Returns:
        user_rating_history ( dict - {string:[(string,string)]} ) -
        {movie_id:[(user_id,rating)]}
    """
    movie_id_to_rating = {} #movie id: [(user id, rating), etc]
    for user in user_rating_history:
        for rating in user_rating_history[user]:
            try:
                movie_id_to_rating[rating[0]].append((user,rating[1]))
            except KeyError:
                movie_id_to_rating[rating[0]] = [(user,rating[1])]
    return movie_id_to_rating

def create_movie_id_to_rating(user_rating_history, movie_id_to_movie):
    movie_id_to_rating = {}  # movie id: [(user id, rating), etc]
    for user in user_rating_history:
        for rating in user_rating_history[user]:
            try:
                movie_id_to_rating[rating[0]].append((user, rating[1]))
            except KeyError:
                movie_id_to_rating[rating[0]] = [(user, rating[1])]
    return movie_id_to_rating
