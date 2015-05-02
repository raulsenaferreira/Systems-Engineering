#=
              LIBS
=#
#Pkg.clone("https://github.com/JuliaDB/DBI.jl.git")
#Pkg.clone("https://github.com/iamed2/PostgreSQL.jl")
#Pkg.clone("https://github.com/JuliaStats/MultivariateStats.jl.git")

#=
              ASSOCIATION RULES
=#

using DBI
using PostgreSQL

macro pr(t)
    :(debugging == true && println($t))
end
debugging = false

function apriori(transactions, threshold, articles)
    combination_max_size = length(articles)
    combs_of_size = [Set(collect(combinations(articles, n))) for n in 1:combination_max_size]

    transactions_count = length(transactions)

    result = {}
    for n=1:combination_max_size
        for comb in combs_of_size[n]
            found_percentage = sum([issubset(comb,t) for t in transactions]) / transactions_count

            if found_percentage < threshold
                for j=n+1:combination_max_size
                    if length(combs_of_size[j]) > 0
                        filter!(c -> !issubset(comb, c), combs_of_size[j])
                    end
                end
            else
                push!(result, comb)
            end
        end
    end
    #@show transactions
    #@show result
    return result
end

function writingResults(resultOfApriori, arrayMovies, filename)
  path=dirname(Base.source_path())
  finalResults={}

  for set in resultOfApriori
    if length(set) > 1
      results={}
      for ind in set
        push!(results, arrayMovies[ind])
      end
      push!(finalResults ,results)
    end
  end
  writedlm("$path/results/$filename", finalResults)
end

function writingClustersResults(info, arrayMovies, filename)
  path=dirname(Base.source_path())
  results = Dict()

  for i=1:length(info)
    if haskey(results, info[i])
      if isdefined(arrayMovies, i)
        results[info[i]]=push!(results[info[i]],arrayMovies[i])
      end
    else
      list={}
      results[info[i]]=push!(list,arrayMovies[i])
    end
  end
  writedlm("$path/results/$filename", results)
end

conn = connect(Postgres, "localhost", "raul", "", "datamining", 5432)

#Movie properties
SQL="select m.id, m.title, g.name from movies m join genres_movies as gm on gm.movie_id=m.id join genres as g on g.id = gm.genre_id;"
stmt = prepare(conn, SQL)

arrayMovies = Array(Dict,1682)
arrayMovies[267]=Dict()#Existe um erro de um registro apenas no banco
result = execute(stmt)
moviesProp = Dict()
for row in result
  if haskey(moviesProp, row[2])
    moviesProp[row[2]]=push!(moviesProp[row[2]],row[3])
  else
    moviesProp = Dict()
    list={}
    moviesProp[row[2]]=push!(list,row[3])
  end
  arrayMovies[row[1]]=moviesProp
end
finish(stmt)

limit=20
#First association rule
subQuery="select movie_id from ratings group by movie_id order by count(movie_id) asc limit $limit"
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
threshold = 0.10
articles=most_freq
transactions=transact

resultOfApriori=apriori(transactions, threshold, articles)
#writing results
writingResults(resultOfApriori, arrayMovies, "rule-1.txt")



#second association rule
subQuery="select movie_id from ratings where rating > 3 group by movie_id order by count(movie_id) desc limit $limit"

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
threshold = 0.2
articles=most_freq
transactions=transact

resultOfApriori=apriori(transactions, threshold, articles)
#writing results
writingResults(resultOfApriori, arrayMovies, "rule-2.txt")



#third association rule
subQuery="select movie_id from ratings where rating < 3 group by movie_id order by count(movie_id) desc limit $limit"

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
threshold = 0.2
articles=most_freq
transactions=transact

resultOfApriori=apriori(transactions, threshold, articles)
#writing results
writingResults(resultOfApriori, arrayMovies, "rule-3.txt")

#Data for clustering
dataSet= zeros(943, 1682)
SQL="select user_id, movie_id from ratings order by user_id;"
stmt = prepare(conn, SQL)
result = execute(stmt)

for row in result
  dataSet[row[1]; row[2]] = 1
end
finish(stmt)
disconnect(conn)

#=
                  CLUSTERING
=#
using Clustering
using MultivariateStats
M=fit(PCA, dataSet; maxoutdim=10)
newMatrix=transform(M, dataSet)

R = kmeans(newMatrix, 10; maxiter=200)
c = counts(R)
# a[i] indicates which cluster the i-th sample is assigned to
a = assignments(R)
#Measuring how well each object lies within its cluster
#silhouettes(R, newMatrix)
writingClustersResults(a, arrayMovies, "clusterResults.txt")


#=
#Reading files (movie lens) if necessary
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
=#
