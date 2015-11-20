#Implementing K-NN and Improved Regularized SVD algorithms for Recommender systems
using DataFrames
using DataArrays

# Initial variables
recomendationType = "item"
K = 10
dataPath = "../Data Mining/Work_1/data/ml-100k/u.data"
dataTable = readtable(dataPath, separator='\t', header=false)
metric = "cosine"


# Functions
function holdOut(base, trainSize)
  R = randperm(size(base, 1))
  indexesTrain = R[1:(size(base, 1)*(trainSize/100))]
  indexesTest = R[(size(indexesTrain, 1)+1):end]
  return base[indexesTrain,:], base[indexesTest,:]
end

function trainMatrix(trainM, matrix, recommendationType)
  for i=1:length(matrix[: , 1])    trainM[matrix[i,1], matrix[i,2]] = matrix[i,3]  end
  trainM = recommendationType=="item" ? trainM : trainM'
  return trainM
end

function testMatrix(similarities, predictions, recomendationType)
  for id in test[:, recomendationType == "item" ? 2 : 1]
    push!(predictions, similarities[id])
  end
  return predictions
end

function metrics(t, u, v)
  if t=="cosine"    return dot(u,v)/(norm(u)*norm(v))
  elseif t=="euclidean"    return 1 - (sqrt(sum((u - v) .^ 2)))
  elseif t=="minkowski"    return 1 - (sum(abs(u - v).^2).^(1/2))
  end
end

function processSimilarities(metric, sims)
  for i = 1:col
    sims[i,i] = -1
    vec1 = trainM[:,i]
    avgs[i] = mean(vec1[find(x -> x != 0, vec1)])
    for j = (i+1):col
      vec2 = trainM[:,j]
      indexes = intersect(find(x -> x != 0, vec1), find(x -> x != 0, vec2))
      sims[j,i] = sims[i,j] = metrics(metric, vec1[indexes], vec2[indexes])
    end
  end
end

function KNNRecommender(K, avgs, simsAvg, similarities)
  for i = 1:col
    simsCol = hcat(simsAvg[:,i], simsAvg[:,1])
    sortedSims = sortrows(simsCol, rev=true)
    similarities[i] = sum(sortedSims[1:K, 1] .* sortedSims[1:K, 2])/sum(sortedSims[1:K, 1])
  end
  return similarities
end

function IRSVDRecommender(predictions, trainM, K, epsilon, lrate, lambda, lambda2)
  u, e, v = svd(DataArray(trainM), K)

  c = zeros(length(unique(users)))
  d = zeros(length(unique(movies)))
  globalMean = mean(train[:,3])

  previous_error = Inf
  errors = zeros(length(train[:,1]))
  mae = mean(abs(errors))
  while (mae > previous_error || (previous_error - mae) < epsilon)
    for i=1:length(train[:,1])
      user = train[i,1]
      item = train[i,2]

      errors[i] = (trainM[user, item]) - (c[user] + d[item] + dot(u[user,:][:,1], v[item,:][:,1]))

      u[user, :] += lrate * (errors[i] * v[item, :] - lambda * u[user, :])
      v[item, :] += lrate * (errors[i] * u[user, :] - lambda * v[item, :])

      c[user] += lrate * (errors[i] - lambda2*(c[user] + d[item] - globalMean))
      d[item] += lrate * (errors[i] - lambda2*(c[user] + d[item] - globalMean))
    end
    mae = mean(abs(errors))
    #if ()
      #stop = true
    #else
      previous_error = mae
    #end
  end

  for i=1:length(test[:,1])
    user = train[i,1]
    item = train[i,2]
    predictions[i] = c[user] + d[item] + dot(u[user,:][:,1], v[item,:][:,1])
  end
end

# Main code... You can use this for KNN and for IRSVD too
users, movies, ratings = array(dataTable)[:,1], array(dataTable)[:,2], array(dataTable)[:,3]

train, test = holdOut(hcat(users, movies, ratings), 80)

trainM = zeros(length(unique(users)), length(unique(movies)))

trainM = trainMatrix(trainM, train, recomendationType)


#preparing use of KNN
col = size(trainM, 2)
sims = zeros(col,col)
avgs = zeros(col)
similarities = zeros(col)

processSimilarities(metric, sims)

simsAvg = hcat(avgs, sims)
simsAvg[find(x -> isnan(x), simsAvg)] = -1

#K-NN
similarities = KNNRecommender(K, avgs, simsAvg, similarities)

predictions = testMatrix(similarities, Float64[], recomendationType)

print (mean(abs(predictions - test[:,3]))) #MAE


#Preparing use of IRSVD
# IRSVD parameters based on this article -> https://www.cs.uic.edu/~liub/KDD-cup-2007/proceedings/Regular-Paterek.pdf
epsilon = 0.001 # minimum error tolerance
lrate = 0.001
lambda = 0.02
lambda2 = 0.05
K = 96
predictions = zeros(length(test[:,1]))
#IRSVD
IRSVDRecommender(predictions, trainM, K, epsilon, lrate, lambda, lambda2)

print (mean(abs(predictions - test[:,3]))) #MAE
