notes = [
    [1, -1, 0, 1, 0, -1],
    [-1, 0, 1, -1, 1, 1],
    [1, 1, 1, 1, -1, -1],
    [1, 1, 0, 0, 1, -1],
    [1, -1, 1, 1, -1, 0]]
nb_users = 5
nb_films = 6
names = ['Alice', 'Bob', 'Charles', 'Daisy', 'Everett']
films = ['007', 'Batman 1', 'Shrek 2', 'Toy Story 3', 'Star Wars 4', 'Twilight 5']
NB_NEIGHBORS = 3
NB_PREDICTIONS = 2

# from big import notes, nb_users, nb_films, names, films, NB_NEIGHBORS, NB_PREDICTIONS

def compute_score(i, j):
    # Your code here
    return score

# print(compute_score(0, 1))  # score(Alice, Bob) = -3
# print(compute_score(0, 2))  # score(Alice, Charles) = 2

def compute_all_scores():
    # Your code here
    return sim

"""for line in compute_all_scores():  # Avoid this line when data is big
    print(line)"""

def nearest_neighbors(i):
    # Your code here
    return neighbors

def compute_prediction(i, i_film, neighbors):
    # Your code here
    return note

def compute_all_predictions(i, neighbors):
    # Your code here
    return candidates[-NB_PREDICTIONS:]

"""print('Predictions for Alice')
neighbors = nearest_neighbors(0)
for line in compute_all_predictions(0, neighbors):
    print(line)"""

def new_user():
    likes = [0] * nb_films
    dislikes = [0] * nb_films
    candidates = []

    for i_film in range(nb_films):
        for i in range(nb_users):
            if notes[i][i_film] == 1:
                likes[i_film] += 1
            elif notes[i][i_film] == -1:
                dislikes[i_film] += 1
        candidates.append((likes[i_film] + dislikes[i_film], films[i_film], i_film))
    candidates.sort()

    mon_id = nb_users
    name = input('Your name? ')
    names.append(name)

    notes.append([0] * nb_films)  # New line in notes
    for _, title, i_film in candidates[-10:]:
        note = int(input('Did you like %s ? (%d notes) ' % (title, _)))
        notes[mon_id][i_film] = note
    return mon_id

mon_id = new_user()
neighbors = nearest_neighbors(mon_id)
for line in compute_all_predictions(mon_id, neighbors):
    print(line)
