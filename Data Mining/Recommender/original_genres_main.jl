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

method = "holdout"

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

arrModel=Array{Recsys.ImprovedRegularedSVD}[]
arrMAE = Array{Float64}[]
experiment = prepareExperiment(method)

for i=1:numGenero
  if method == "holdout"
    model, arrTest = executeRecommenderHoldOut(experiment, i, d)
    if isempty(arrModel)
      arrModel = model
    else
      arrModel = hcat(arrModel, model)
    end
  elseif method == "kfold"
    model, maes = executeRecommenderKFold(experiment, i, d)
    if isempty(arrMAE)
      arrMAE = maes
      arrModel = model
    else
      arrMAE = hcat(arrMAE, maes)
      arrModel = hcat(arrModel, model)
    end
  end
end

if method == "kfold"
  trainMAEs = Float64[]
  for i=1:length(arrMAE[:,1])
    m = mean(arrMAE[:,i])
    push!(trainMAEs, m)
  end

  index = indmin(trainMAEs)
  models = arrModel[index,:]
  arrMAE = models

  test_data = experiment.getTestData(index)
end

targets     = Int64[]
predictions = Float64[]

if method == "kfold"
  for i=1:length(test_data[:,1])
    genres = d[test_data[i,2]]
    sum = 0.0
    total = 0.0
    for genre in genres
      if genre != 19
        model = arrModel[genre]
        one_element = [test_data[i,1] test_data[i,2]]
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
elseif method == "holdout"
  offset = 4
  for k=1:10
    test_data = arrTest[:,(k-1)*offset + 1:(k-1)*offset + offset]
    models = arrModel[k,:]
    for i=1:length(test_data[:,1])
      genres = d[test_data[i,2]]
      sum = 0.0
      total = 0.0
      for genre in genres
        if genre != 19
          model = models[genre]
          one_element = [test_data[i,1] test_data[i,2]]
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

function prepareExperiment(method)
  data = Recsys.Dataset();
  if method == "kfold"
    experiment = Recsys.KFold(10);
  elseif method == "holdout"
    experiment = Recsys.HoldOut(0.9, data);
  end
  return experiment
end

function executeRecommenderHoldOut(experiment, genre, item_genres)
  data = Recsys.Dataset();

  arrModel=Recsys.ImprovedRegularedSVD[]
  arrTest = Array{Float64}[]
  for i=1:10
    experiment = Recsys.HoldOut(0.9, data);

    train_data = experiment.getTrainData();
    test_data = experiment.getTestData();

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

    push!(arrModel, model)
    if isempty(arrTest)
      arrTest = test_data
    else
      arrTest = hcat(arrTest, test_data)
    end
  end

  return arrModel, arrTest[2:length(arrTest[:,1]),:]
end

function executeRecommenderKFold(experiment, genre, item_genres)
  data = Recsys.Dataset();

  trainMAEs = Float64[]
  arrModel = Recsys.ImprovedRegularedSVD[]
  kfold = 10
  for i=1:kfold
    train_data = experiment.getTrainData(i);
    #test_data = experiment.getTestData(i);

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

    genre_train_data = genre_train_data.file
    genre_train_data = [genre_train_data[:,1] genre_train_data[:,2] genre_train_data[:,3] genre_train_data[:,4]]
    predictions = model.predict(genre_train_data[:,1:2]);

    MAE = Recsys.mae(predictions, genre_train_data[:,3])

    push!(arrModel, model)
    push!(trainMAEs, MAE)
  end

  return arrModel, trainMAEs
end
