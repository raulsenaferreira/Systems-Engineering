#=
Pkg.clone("https://github.com/JuliaDB/DBI.jl.git")
Pkg.clone("https://github.com/iamed2/PostgreSQL.jl")
Pkg.clone("https://github.com/JuliaStats/MultivariateStats.jl.git")
Pkg.clone("https://github.com/JuliaStats/Clustering.jl")
Pkg.clone("https://github.com/filipebraida/Recsys.jl.git")
=#

using DBI
using PostgreSQL
using Recsys
using Clustering
#reload("Recsys")

isGrupo = false
numGenero = 18
arrayGenero = [1:numGenero]
path=dirname(Base.source_path())

SQLGenerateByGenreNumber(arrayGenero, isGrupo)

results = "-results"
#file=string("$path/original/all_results-group-", isGrupo)
file = "$path/original"

arrTests=Array{Int64}[]
arrModel=Recsys.ImprovedRegularedSVD[]
arrRatings=Array{Int64}[]

for i=1:numGenero
  filename = string(file, "/gender-$i")

  model, test_data, ratings = executeRecommender(filename)

  push!(arrModel, model)
  push!(arrTests, test_data)
  push!(arrRatings, ratings)
end

arrTests
arrModel
arrRatings

SQL = "SELECT movie_id, array_agg(genre_id)::text AS genres
        FROM genres_movies
        GROUP BY movie_id
        ORDER BY movie_id"

conn = connect(Postgres, "localhost", "postgres", "postgres", "movielens", 5432)
stmt = prepare(conn, SQL)
result = execute(stmt)

d = Dict()
for row in result
  indexes = replace(row[2], "{", "")
  indexes = replace(indexes, "}", "")
  d[int(row[1])] = int(split(indexes, ","))
end

finish(stmt)
disconnect(conn)

targets     = Int64[]
predictions = Float64[]

predictedIds= Array{Int64}[]

for (arrTest, arrRating) in zip(arrTests, arrRatings)
  for i=1:length(arrTest[:,1])
    if !findElement(predictedIds, [arrTest[i,1] arrTest[i,2]])
      genres = d[arrTest[i,2]]
      sum = 0.0
      total = 0.0
      for genre in genres
        if genre != 19
          model = arrModel[genre]
          one_element = [arrTest[i,1] arrTest[i,2]]
          sum += model.predict(one_element)
          total += 1
        end
      end

      predicted_rating = sum/total
      original_rating  = arrRating[i]
      push!(predictions, round(predicted_rating[1], 3))
      push!(targets, original_rating)
      push!(predictedIds, [arrTest[i,1] arrTest[i,2]])
    end
  end
end

predictions = reshape(predictions, length(predictedIds), 1)
targets

MAE = Recsys.mae(predictions, targets)
RMSE = Recsys.rmse(predictions, targets)

  #=
  trainData = experiment.getTrainData(i).file

  trainData = trainData[find(r->in(r, index),trainData[:item]),:]

  datasetCluster = Recsys.Dataset(trainData, dataset.users, dataset.items, dataset.preferences)

  model = Recsys.ImprovedRegularedSVD(datasetCluster, 10);
  push!(arrModel, arrModel)
  testData = experiment.getTestData(i)[:,1:2]

  testData = testData[find(r->in(r, index),testData[:,2]),:]
  push!(arr, arrayTestData)
  predictions = model.predict(testData);
  =#

function SQLGenerateByGenreNumber(arrayGenero, isGrupo)
  SQL="SELECT	r.user_id, r.movie_id, r.rating
            FROM ratings as r "
  sqli=""
  genres=""
  if isGrupo
    for i in arrayGenero
      sqlit=" join genres_movies as gm$i on gm$i.movie_id = r.movie_id
      join genres as g$i on g$i.id = gm$i.genre_id
      AND g$i.id = $i"
      sqli=string(sqli, sqlit)
      genres=string(genres, "-$i")
    end

    SQL = string(SQL, sqli, "ORDER BY r.user_id;")
    filename=string("genres", genres)
    clusteringCreate(SQL, filename)
  #Todos os filmes
  else
    for i in arrayGenero
      SQL="SELECT	r.user_id, r.movie_id, r.rating
            FROM ratings as r
            join genres_movies as gm on gm.movie_id = r.movie_id
            join genres as g on g.id = gm.genre_id
      WHERE g.id = $i
      ORDER BY r.user_id;"

      filename="gender-$i"
      clusteringCreate(SQL, filename)
    end
  end
end


# Cria cluster com os gêneros originais e salva em arquivo
function clusteringCreate(SQL, filename)
  path=dirname(Base.source_path())
  conn = connect(Postgres, "localhost", "raul", "", "movielens", 5432)
  stmt = prepare(conn, SQL)
  result = execute(stmt)

  tuple = int(zeros(4)')
  for row in result
    tuple = vcat(tuple, [int(row[1]), int(row[2]), int(row[3]), 0]')
  end
  finish(stmt)
  disconnect(conn)
  writedlm("$path/original/$filename", tuple[2:length(tuple[:,1]),:])
end

function executeRecommender(file)
  data = Recsys.Dataset(file);
  experiment = Recsys.KFold(10);
  train_data = experiment.getTrainData(1);
  test_data = experiment.getTestData(1);
  model = Recsys.ImprovedRegularedSVD(train_data, 10)

  return model, test_data[:,1:2], test_data[:,3]
end

function findElement(A, pair)
  element1 = pair[1]
  element2 = pair[2]
  for a in A
    if a[1] == element1 && a[2] == element2
      return true
    end
  end
  return false
end

# Test
path=dirname(Base.source_path())
data = Recsys.Dataset("$path/original/gender-1")
experiment = Recsys.KFold(10);
train_data = experiment.getTrainData(1);
test_data = experiment.getTestData(1);

model = Recsys.ImprovedRegularedSVD(train_data, 10)

predictions = model.predict(test_data);
targets = test_data[:,3]

MAE = Recsys.mae(predictions, targets)
RMSE = Recsys.rmse(predictions, targets)
