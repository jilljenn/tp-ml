import csv

nb_users = 668
nb_films = 9800
notes = [[0] * nb_films for _ in range(nb_users)]
names = list(range(nb_users))
films = [''] * nb_films
NB_NEIGHBORS = 50
NB_PREDICTIONS = 20

with open('ratings-ml.csv', newline='') as csvfile:
    f = csv.reader(csvfile)
    nb_users = 0
    nb_films = 0
    nb_notes = 0
    for ligne in f:
        i, i_film, note = ligne
        i = int(i)
        i_film = int(i_film)
        note = int(note)
        notes[i][i_film] = note
        nb_users = max(i + 1, nb_users)
        nb_films = max(i_film + 1, nb_films)
        nb_notes += 1
    print(nb_notes, 'notes loaded from', nb_users, 'users over', nb_films, 'films')

with open('works-ml.csv', newline='') as csvfile:
    f = csv.reader(csvfile)
    for ligne in f:
        i_film, titre = ligne
        films[int(i_film)] = titre
