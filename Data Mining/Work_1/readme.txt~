The objective is mining data in movielens database and thus, answer following questions:
1) Who watches movies X watches Y.
2) Who like movies X likes  Y.
3) Who dislikes movies X dislikes Y.
4) Who dislikes X but likes Y.(Try to find a interesting pattern)
P.S.: From the questions above, find 10 most frequent patterns.
This work uses SVD or PCA to reduce dimensionality data(about 5 or 10 collumns) and K-Means or DB Scan to clustering groups that watched the same movies(about 10 and 15 clusters).
For learning and personal challenge purposes this work is under development using julia language.

package dependencies:
https://github.com/JuliaDB/DBI.jl
https://github.com/iamed2/PostgreSQL.jl
https://github.com/jonathanewerner/julia-apriori-algorithm
https://github.com/JuliaStats/Clustering.jl


Queries:
1)Association rules for "which movies(ids) were watched by each user(id) and which are their genres":
select r.user_id, array_agg(r.movie_id), array_agg(gm.genre_id) from ratings r
join genres_movies as gm on r.movie_id = gm.movie_id
group by r.user_id

I assume a support (40%) to calculate the apriori algorithm because the frequence of films watched aren't equal.

select user_id, movie_id from ratings where movie_id in (
  select movie_id from ratings group by movie_id order by count(movie_id) desc limit 20
) group by user_id, movie_id order by user_id;

Only the 20 most relevant films.
Results: result => {{294},{174},{117},{258},{121},{1},{288},{100},{237},{300},{127},{50},{7},{56},{181},{286},{98},{50,1},{50,100},{50,181},{50,174}}


2)liked X and liked Y
select movie_id from ratings where rating > 3 group by movie_id order by count(movie_id) desc 
results:result => {{174},{258},{1},{288},{100},{237},{300},{127},{50},{7},{56},{181},{286},{98},{50,1},{50,100},{50,181},{50,174}}



3)Disliked X, disliked Y
select movie_id from ratings where rating < 3 group by movie_id order by count(movie_id) desc
results: result => {{294},{121},{288},{286}}
