Pkg.clone("https://github.com/JuliaDB/DBI.jl.git")
Pkg.clone("https://github.com/iamed2/PostgreSQL.jl")
Pkg.clone("https://github.com/JuliaStats/MultivariateStats.jl.git")
Pkg.clone("https://github.com/JuliaStats/Clustering.jl")

using DBI
using PostgreSQL
using Recsys
reload("Recsys")

conn = connect(Postgres, "localhost", "raul", "", "movielens", 5432)
dataSet= zeros(1682, 943)

#=
                  CLUSTERING
=#

#Data for clustering
SQL="select movie_id, user_id, rating from ratings;"
stmt = prepare(conn, SQL)
result = execute(stmt)

for row in result
  dataSet[row[1]; row[2]] = row[3]
end
originalDataSet = copy(dataSet)
medias=zeros(1682)
for i=1:length(dataSet[:,1])
  cont=0
  soma=0
  for j=1:length(dataSet[1,:])
    if dataSet[i;j] != 0
      soma += dataSet[i;j]
      cont+=1
    end
  end
  medias[i]=soma/cont;
end
for i=1:length(dataSet[:,1])
  for j=1:length(dataSet[1,:])
    dataSet[i;j]=dataSet[i;j]-medias[i]
  end
end
finish(stmt)
disconnect(conn)

#=
                  CLUSTERING
=#
using Clustering
using MultivariateStats
M=fit(PCA, dataSet'; maxoutdim=10)
newMatrix=transform(M, dataSet')

numCluster = 25
R = kmeans(newMatrix, numCluster; maxiter=200)
c = counts(R)
# a[i] indicates which cluster the i-th sample is assigned to
a = assignments(R)

matrizClusters = [originalDataSet', a']'

#=
function writingClustersResults(numCluster, matriz)
  path=dirname(Base.source_path())

  for k=1:numCluster
    aux=zeros(3)
    filename="cluster-$k"
    for i=1:length(matriz[:,1])
      if matriz[i,length(matriz[1,:])]==k
        for j=1:length(matriz[1,:])
          aux = vcat(aux,[i, j, matriz[i,j]])
        end
      end
    end
    writedlm("$path/results/$filename", aux[2:length(aux[:,1]),:])
  end
end
writingClustersResults(10, originalDataSet) =#

path=dirname(Base.source_path())
matriz = matrizClusters

for k=1:numCluster
  aux=int(zeros(4)')
  filename="cluster-$k"
  for i=1:length(matriz[:,1])
    if matriz[i,length(matriz[1,:])]==k
      for j=1:length(matriz[1,1:943])
        if matriz[i,j] != 0
          temp = int(zeros(4)')

          temp[1] = int(j)
          temp[2] = int(i)
          temp[3] = int(matriz[i,j])
          temp[4] = 0

          aux = vcat(aux, temp)
        end
      end
    end
  end
  #writedlm("$path/results/$filename", aux)
  writedlm("$path/results/$filename", aux[2:length(aux[:,1]),:])
end

function executeRecommender(file)
  data = Recsys.Dataset(file);

  experiment = Recsys.HoldOut(0.9, data);
  train_data = experiment.getTrainData();
  test_data = experiment.getTestData();

  model01 = Recsys.ImprovedRegularedSVD(train_data, 10);
  predictionsSVD = model01.predict(test_data[:,1:2]);

  #m1MAE = Recsys.mae(predictionsSVD, test_data[:,3]);
  #m1RMSE = Recsys.rmse(predictionsSVD, test_data[:,3]);

  return predictionsSVD#m1MAE, m1RMSE
end

function executeRecommenderKFold(file, kfold)
  resultsPredictions=zeros(1)
  testDataAll = zero(1)
  for i=1:kfold
    data = Recsys.Dataset(file);

    experiment = Recsys.HoldOut(0.9, data);
    train_data = experiment.getTrainData();
    test_data = experiment.getTestData();

    model01 = Recsys.ImprovedRegularedSVD(train_data, 10);
    predictionsSVD = model01.predict(test_data[:,1:2]);

    resultsPredictions = vcat(resultsPredictions, predictionsSVD[:,1])
    testDataAll = vcat(testDataAll, test_data[:,3])
  end
  return resultsPredictions[2:length(resultsPredictions)], testDataAll[2:length(testDataAll)]
end

#= Rodando individual
k = 10
filename="cluster-$k"

MAE, RMSE = executeRecommender("$path/results/$filename")
 End =#

#=
  Rodando o IRSVD para cada um dos clusteres criados
=#
kfold = 10
resultsPredictionsTotal = zeros(1)
testDataTotal = zeros(1)

for k=1:numCluster
  resultsMAE = zeros(kfold)
  resultsRMSE = zeros(kfold)
  resultsPredictions=zeros(kfold)

  filename="cluster-$k"
  results = "-results"

  resultsPredictions, testData = executeRecommenderKFold("$path/results/$filename", kfold)

  writedlm("$path/results/$filename$results", [Recsys.mae(resultsPredictions, testData), Recsys.rmse(resultsPredictions, testData)])

  resultsPredictionsTotal = vcat(resultsPredictionsTotal, resultsPredictions)
  testDataTotal = vcat(testDataTotal, testData)

end

MAE = Recsys.mae(resultsPredictionsTotal[2:length(resultsPredictionsTotal)], testDataTotal[2:length(testDataTotal)])
RMSE = Recsys.rmse(resultsPredictionsTotal[2:length(resultsPredictionsTotal)], testDataTotal[2:length(testDataTotal)])

writedlm("$path/results/all_results", [MAE, RMSE])

#=
  Rodando o IRSVD para o dataset original
=#
data = Recsys.Dataset();

resultsMAE = zeros(10)
resultsRMSE = zeros(10)
for i=1:10
  #Separando os dados em treino e teste
  experiment = Recsys.HoldOut(0.9, data);
  train_data = experiment.getTrainData();
  test_data = experiment.getTestData();

  #Treinos
  model01 = Recsys.ImprovedRegularedSVD(train_data, 10);

  #Teste
  predictionsSVD = model01.predict(test_data[:,1:2]);

  m1MAE = Recsys.mae(predictionsSVD, test_data[:,3]);
  m1RMSE = Recsys.rmse(predictionsSVD, test_data[:,3]);

  resultsMAE[i] = m1MAE
  resultsRMSE[i] = m1RMSE
end

filename="original"
results = "-results"

writedlm("$path/results/$filename$results", [m1MAE, m1RMSE])

#=
  Preparação para dividir e rodar pelos gêneros originais
=#

conn = connect(Postgres, "localhost", "raul", "", "movielens", 5432)

numGenero = 18

for genero=1:numGenero
  #Data for clustering
  SQL = "SELECT
          ratings.user_id,
          movies.id AS movie_id,
          ratings.rating
    FROM movies, genres_movies, genres, ratings
    WHERE movies.id = genres_movies.movie_id
    AND genres.id = genres_movies.genre_id
    AND ratings.movie_id = movies.id
    AND genres.id = $genero
   ORDER BY ratings.user_id;"

  stmt = prepare(conn, SQL)
  result = execute(stmt)

  tuple = int(zeros(4)')
  for row in result
    tuple = vcat(tuple, [int(row[1]), int(row[2]), int(row[3]), 0]')
  end

  filename="gender-$genero"
  writedlm("$path/results/original/$filename", tuple[2:length(tuple[:,1]),:])
end

finish(stmt)
disconnect(conn)

resultsPredictionsTotal = zeros(1)
testDataTotal = zeros(1)

for k=1:numGenero
  resultsMAE = zeros(kfold)
  resultsRMSE = zeros(kfold)
  resultsPredictions=zeros(kfold)

  filename="gender-$k"
  results = "-results"

  resultsPredictions, testData = executeRecommenderKFold("$path/results/original/$filename", kfold)

  writedlm("$path/results/original/$filename$results", [Recsys.mae(resultsPredictions, testData), Recsys.rmse(resultsPredictions, testData)])

  resultsPredictionsTotal = vcat(resultsPredictionsTotal, resultsPredictions)
  testDataTotal = vcat(testDataTotal, testData)

end

MAE = Recsys.mae(resultsPredictionsTotal[2:length(resultsPredictionsTotal)], testDataTotal[2:length(testDataTotal)])
RMSE = Recsys.rmse(resultsPredictionsTotal[2:length(resultsPredictionsTotal)], testDataTotal[2:length(testDataTotal)])

writedlm("$path/results/original/all_results", [MAE, RMSE])
#=
for genero=1:numGenero
  resultsMAE = zeros(numGenero)
  resultsRMSE = zeros(numGenero)
  filename="gender-$genero"
  results = "-results"
  for i=1:10
    MAE, RMSE = executeRecommender("$path/results/original//$filename")
    resultsMAE[i] = MAE
    resultsRMSE[i] = RMSE
  end
  writedlm("$path/results/original/$filename$results", [mean(resultsMAE), mean(resultsRMSE)])
end
=#
#=
experiment = Recsys.KFold(10);

function model(data)
  return Recsys.ImprovedRegularedSVD(data);
end

result = experiment.getTrainData()#run(model, 1)

@show mean(result[:mae])

=#

# Algoritmo original
#-------------------------------------------------------------------------------------------------------
data = Recsys.Dataset();

#Separando os dados em treino e teste
experiment = Recsys.HoldOut(0.9, data);
train_data = experiment.getTrainData();
test_data = experiment.getTestData();


#Treinos
model01 = Recsys.ImprovedRegularedSVD(train_data, 10);
#model02 = Recsys.KNN(train_data);
#model04 = Recsys.GlobalMean(train_data);
#model03 = Recsys.UserMean(train_data);

#Teste
predictionsSVD = model01.predict(test_data[:,1:2]);
#predictionsKNN = model02.predict(test_data[:,1:2]);
#predictionsGM = model04.predict(test_data[:,1:2]);
#predictionsUM = model03.predict(test_data[:,1:2]);
#predictionsSVD = predictionsSVD[:,1]
#Combinando os resultados
#resultado = mean(predictionsSVD, predictionsUM, predictionsGM);
#resultado = mean([predictionsSVD, predictionsUM, predictionsGM]);

#Calculando os erros combinados e separados
#globalMAE = Recsys.mae(resultado, test_data[:,3]);

#@show globalMAE

m1MAE = Recsys.mae(predictionsSVD, test_data[:,3]);
m1RMSE = Recsys.rmse(predictionsSVD, test_data[:,3]);

filename="original"
results = "-results"
writedlm("$path/results/$filename$results", [m1MAE, m1RMSE])

@show m1MAE

m3MAE = Recsys.mae(predictionsUM, test_data[:,3]);







# ----------------------Codigo Filipe-----------------------
using Recsys
using Clustering

experiment = Recsys.KFold(10);

dataset = experiment.getTrainData(1)

matrix = experiment.getTrainData(1).getMatrix();

(U, S, V) = svd(full(matrix));

itemFeatures = V[:,1:10];

numCluster = 10
cluster = kmeans(itemFeatures', numCluster; maxiter=200)
c = counts(cluster)
a = assignments(cluster)

for i=1:numCluster
  index = find(r->r==i, a)

  trainData = experiment.getTrainData(1).file

  trainData = trainData[find(r->in(r, index),trainData[:item]),:]

  datasetCluster = Recsys.Dataset(trainData, dataset.users, dataset.items, dataset.preferences)



  model = Recsys.ImprovedRegularedSVD(datasetCluster, 10);

  testData = experiment.getTestData(1)[:,1:2]

  testData = testData[find(r->in(r, index),testData[:,2]),:]

  predictions = model.predict(testData);
end


