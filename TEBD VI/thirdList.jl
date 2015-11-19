#Implementing K-NN and Improved Regularized SVD algorithms for Recommender systems

#K-NN
using DataFrames

# Initial variables
recomendationType = "item"
K = 10
dataPath = "/home/filipebraida/workspace/Systems-Engineering/u.data"
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
      #sims[i,j] = metrics(metric, vec1[indexes], vec2[indexes])
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


# Main code
users, movies, ratings = array(dataTable)[:,1], array(dataTable)[:,2], array(dataTable)[:,3]

train, test = holdOut(hcat(users, movies, ratings), 80)

trainM = zeros(length(unique(users)), length(unique(movies)))

trainM = trainMatrix(trainM, train, recomendationType)

col = size(trainM, 2)
sims = zeros(col,col)
avgs = zeros(col)
similarities = zeros(col)

processSimilarities(metric, sims)

simsAvg = hcat(avgs, sims)
simsAvg[find(x -> isnan(x), simsAvg)] = -1

similarities = KNNRecommender(K, avgs, sims_avg, similarities)

predictions = testMatrix(similarities, Float64[], recomendationType)

print (mean(abs(predictions - test[:,3]))) #MAE




#IRSVD
