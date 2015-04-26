path=dirname(Base.source_path());
# u.item     -- Information about the items (movies)
movies = readdlm("$path/data/ml-100k/u.item", '|');

#u.data     -- 100000 ratings by 943 users on 1682 items
prefs = readdlm("$path/data/ml-100k/u.data", '\t');
#data1=prefs[:,1:2]

#u.user     -- Demographic information about the users
users = readdlm("$path/data/ml-100k/u.user", '|');

#print (movies[1]);
#print (prefs[1]);
#print (users[1]);



#=
              ASSOCIATION RULES
=#
#Pkg.clone("https://github.com/JuliaDB/DBI.jl.git")
#Pkg.clone("https://github.com/iamed2/PostgreSQL.jl")
using DBI
using PostgreSQL

macro pr(t)
    :(debugging == true && println($t))
end
debugging = true

function apriori(transactions, threshold, articles)
    combination_max_size = length(articles)
    # combs_of_size[n] = all possible article combinations of size n
    combs_of_size = [Set(collect(combinations(articles, n))) for n in 1:combination_max_size]

    transactions_count = length(transactions)

    result = {}
    for n=1:combination_max_size
       #@pr "------------------ looking at all combinations of size $n (combs_of_size[n]) -----------------"
       #@pr "those are $(combs_of_size[n])\n"

        for comb in combs_of_size[n]
            # the percentage of transactions which the combination (comb) is a subset of
            # (e.g.: [1,2] is subset of t3=[1,2] or of t2=[1,2,4])
            found_percentage = sum([issubset(comb,t) for t in transactions]) / transactions_count

            if found_percentage < threshold
                #@pr "$comb is not in $(threshold*100)% of the transactions, thus "
                    #"do not add it to result and possibly prune other bigger combos containing it."

                for j=n+1:combination_max_size
                    if length(combs_of_size[j]) > 0
                        #@pr "\tpruning combs_of_size[$j] (= all combos of size $j)"
                        #@pr "\t\tbefore: $(combs_of_size[j])"
                        filter!(c -> !issubset(comb, c), combs_of_size[j])
                        #@pr "\t\tafter: $(combs_of_size[j])"
                    end
                end
            else
                #@pr "$comb is in $(threshold*100)% of the transactions, thus adding it to result."
                push!(result, comb)
            end
        end
        #@pr "\nfinished iteration for combinations of size $n, current result is:\n$result"

    end
    #@show transactions
    @show result
end



conn = connect(Postgres, "localhost", "raul", "", "datamining", 5432)
limit=50

#First association rule
subQuery="select movie_id from ratings group by movie_id order by count(movie_id) asc limit $limit"

stmt = prepare(conn, subQuery)
result=execute(stmt)
most_freq={}
for row in result
  push!(most_freq, row[1])
end
finish(stmt)

#stmt = prepare(conn, "select array_agg(distinct(r.movie_id))::varchar from ratings r join genres_movies as gm on r.movie_id = gm.movie_id group by r.user_id")
SQL="select user_id, movie_id from ratings where movie_id in ( $subQuery ) group by user_id, movie_id order by user_id;"
stmt = prepare(conn, SQL)
movies = Dict()
result = execute(stmt)
for row in result
  if haskey(movies, row[1])
    movies[row[1]]=hcat(movies[row[1]],row[2])
  else
    movies[row[1]]=row[2]
  end
end
finish(stmt)

transact={}
for v = values(movies)
  push!(transact, v)
end
threshold = 0.3
articles=most_freq
transactions=transact'

apriori(transactions, threshold, articles)



#second association rule
subQuery="select movie_id from ratings where rating > 3 group by movie_id order by count(movie_id) asc limit $limit"

stmt = prepare(conn, subQuery)
result=execute(stmt)
most_freq={}
for row in result
  push!(most_freq, row[1])
end
finish(stmt)

SQL="select user_id, movie_id from ratings where movie_id in ( $subQuery ) group by user_id, movie_id order by user_id;"
stmt = prepare(conn, SQL)
movies = Dict()
result = execute(stmt)
for row in result
  if haskey(movies, row[1])
    movies[row[1]]=hcat(movies[row[1]],row[2])
  else
    movies[row[1]]=row[2]
  end
end
finish(stmt)

transact={}
for v = values(movies)
  push!(transact, v)
end
threshold = 0.4
articles=most_freq
transactions=transact'

apriori(transactions, threshold, articles)



#third association rule
subQuery="select movie_id from ratings where rating < 3 group by movie_id order by count(movie_id) asc limit $limit"

stmt = prepare(conn, subQuery)
result=execute(stmt)
most_freq={}
for row in result
  push!(most_freq, row[1])
end
finish(stmt)

SQL="select user_id, movie_id from ratings where movie_id in ( $subQuery ) group by user_id, movie_id order by user_id;"
stmt = prepare(conn, SQL)
movies = Dict()
result = execute(stmt)
for row in result
  if haskey(movies, row[1])
    movies[row[1]]=hcat(movies[row[1]],row[2])
  else
    movies[row[1]]=row[2]
  end
end
finish(stmt)

transact={}
for v = values(movies)
  push!(transact, v)
end
threshold = 0.4
articles=most_freq
transactions=transact'

apriori(transactions, threshold, articles)

disconnect(conn)



#=
                  CLUSTERING
=#
using Clustering

R = kmeans(data1', 10; maxiter=200)
@assert nclusters(R) == 10
# a[i] indicates which cluster the i-th sample is assigned to
a = assignments(R)
# c[k] is the number of samples assigned to the k-th cluster
c = counts(R)
# M[:,k] is the mean vector of the k-th cluster
M = R.centers
