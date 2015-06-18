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

numGenero = 18
arrayGenero = [1:numGenero]
path=dirname(Base.source_path())

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

arrModel=Recsys.ImprovedRegularedSVD[]

for i=1:numGenero
  experiment = prepareExperiment()
  model = executeRecommender(experiment, i, d)

  push!(arrModel, model)
end

test_data = experiment.getTestData(1)

targets     = Int64[]
predictions = Float64[]

for i=1:length(test_data[:,1])
  genres = d[test_data[i,2]]
  sum = 0.0
  total = 0.0
  for genre in genres
    if genre != 19
      model = arrModel[genre]
      one_element = [test_data[i,1] test_data[i,2]]
      println(test_data[i,:])
      sum += model.predict(one_element)
      total += 1
    end
  end
  if sum != 0.0 && total != 0.0
    predicted_rating = sum/total
    original_rating  = test_data[i,3]
    push!(predictions, predicted_rating[1])
    push!(targets, original_rating)
  end
end

predictions = reshape(predictions, length(predictions), 1)
targets

writedlm("$path/predictions.txt", predictions)
writedlm("$path/targets.txt", targets)

MAE = Recsys.mae(predictions, targets)
RMSE = Recsys.rmse(predictions, targets)

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

function prepareExperiment()
  data = Recsys.Dataset();
  experiment = Recsys.KFold(10);
  return experiment
end

function executeRecommender(experiment, genre, item_genres)

  train_data = experiment.getTrainData(1);
  test_data = experiment.getTestData(1);

  item_data = train_data.file

  # Gambis, descobrir como faz corretamente
  genre_train_data = item_data[1,:]
  for genres in item_genres
    if in(genre, genres[2])
      itens = find(x -> x == genre[1], item_data[2])
      selected_data = [ [item_data[i,1] item_data[i,2] item_data[i,3] item_data[i,4]] for i in itens ]
      for sd in selected_data
        push!(genre_train_data, sd)
      end
    end
  end
  genre_train_data = Recsys.Dataset(genre_train_data[2:length(genre_train_data),:], data.users, data.items, data.preferences)
  model = Recsys.ImprovedRegularedSVD(genre_train_data, 10)

  return model
end
