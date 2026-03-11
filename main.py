import numpy as np
import pandas as pd
from flask import Flask, render_template, request
# libraries for making count matrix and similarity matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# define a function that creates similarity matrix
# if it doesn't exist
def create_sim():
    data = pd.read_csv('data.csv')
    data['movie_title'] = data['movie_title'].astype(str).str.lower()
    data['comb'] = data['comb'].astype(str)
    # creating a count matrix
    cv = TfidfVectorizer()
    count_matrix = cv.fit_transform(data['comb'])
    # creating a similarity score matrix
    sim = cosine_similarity(count_matrix)
    return data, sim


data, sim = create_sim()

# defining a function that recommends 10 most similar movies
def rcmd(m):
    if not m:
        return 'This movie is not in our database.\nPlease check if you spelled it correct.'
    
    m = m.lower().strip()
    
    # check if data and sim are already assigned
    try:
        data.head()
        sim.shape
    except:
        pass

    # check if the movie is in our database or not
    if m not in data['movie_title'].unique():
        return 'This movie is not in our database.\nPlease check if you spelled it correct.'
    else:
        # getting the index of the movie in the dataframe
        i = data.loc[data['movie_title'] == m].index[0]

        # fetching the row containing similarity scores of the movie
        # from similarity matrix and enumerate it
        lst = list(enumerate(sim[i]))

        # sorting this list in decreasing order based on the similarity score
        lst = sorted(lst, key=lambda x: x[1], reverse=True)

        # taking top 1- movie scores
        # not taking the first index since it is the same movie
        lst = lst[1:11]

        # making an empty list that will containg all 10 movie recommendations
        l = []
        for i in range(len(lst)):
            a = lst[i][0]
            l.append(data['movie_title'][a])
        return l

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/recommend")
def recommend():
    movie = request.args.get('movie')
    r = rcmd(movie)
    movie = movie.title() if movie else "Unknown"
    if type(r) == type('string'):
        return render_template('recommend.html', movie=movie, r=r, t='s')
    else:
        return render_template('recommend.html', movie=movie, r=r, t='l')

@app.route("/movies")
def movies():
    search = request.args.get('search', '').strip().lower()
    selected_genre = request.args.get('genre', '').strip().lower()
    page = request.args.get('page', 1, type=int)
    per_page = 12

    movies_data = data[['movie_title', 'genres', 'actor_1_name', 'actor_2_name', 'actor_3_name', 'director_name']].copy()
    movies_data = movies_data.fillna('Unknown')

    all_genres = set()
    for genres in movies_data['genres']:
        for genre in str(genres).split('|'):
            genre = genre.strip()
            if genre:
                all_genres.add(genre.title())

    if search:
        movies_data = movies_data[movies_data['movie_title'].str.contains(search, case=False, na=False)]

    if selected_genre:
        movies_data = movies_data[movies_data['genres'].str.lower().str.contains(selected_genre, na=False)]

    movie_list = []
    for _, row in movies_data.iterrows():
        movie_list.append({
            'title': row['movie_title'].title(),
            'genres': str(row['genres']).replace('|', ', '),
            'actor_1': row['actor_1_name'],
            'actor_2': row['actor_2_name'],
            'actor_3': row['actor_3_name'],
            'director': row['director_name']
        })

    total_movies = len(movie_list)
    total_pages = (total_movies + per_page - 1) // per_page

    if page < 1:
        page = 1
    if total_pages > 0 and page > total_pages:
        page = total_pages

    start = (page - 1) * per_page
    end = start + per_page
    paginated_movies = movie_list[start:end]

    return render_template(
        'movies.html',
        movies=paginated_movies,
        genres=sorted(all_genres),
        search=search,
        selected_genre=selected_genre,
        page=page,
        total_pages=total_pages,
        total_movies=total_movies
    )

if __name__ == '__main__':
    app.run(debug=True)