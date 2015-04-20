path=dirname(Base.source_path());
#=
u.item     -- Information about the items (movies)
movie id | movie title | release date | video release date | IMDb URL | unknown | Action | Adventure | Animation |
Children's | Comedy | Crime | Documentary | Drama | Fantasy | Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi |
Thriller | War | Western |
=#
movies = readdlm("$path/data/ml-100k/u.item", '|');
#=
u.data     -- 100000 ratings by 943 users on 1682 items
user id | item id | rating | timestamp
=#
prefs = readdlm("$path/data/ml-100k/u.data", '\t');
#=
u.user     -- Demographic information about the users
user id | age | gender | occupation | zip code
=#
users = readdlm("$path/data/ml-100k/u.user", '|');
#print (movies[1]);
#print (prefs[1]);
#print (users[1]);

#using ODBC
#connection = advancedconnect("Driver={psqlODBC};user=raul;server=localhost;database=datamining;")
