using Gadfly

#Question 1
array = readdlm("u.data")
qtyPerUser = hcat(hist(array[:,1], 0.1:943.9)[2], 1:943)
qtyPerUser = sortrows(qtyPerUser,by=x->(x[1]), rev=true)
plot(x=1:size(qtyPerUser)[1], y=qtyPerUser[:,1], Geom.bar)


#Question 2
plot(x=1:5, y=hist(array[:,3], 0:5.9)[2], Geom.bar)


#Question 3
numLines=size(array, 1)
randomBase = array[randperm(numLines), :]
training = randomBase[1:(numLines*0.8), :]
test = randomBase[(numLines*0.8+1):end, :]

#media dos ratings
globalMean = mean(training[:,3])
userAverages = [mean(training[find(x->(x==i), training[:,1]), 3]) for i=1:943]

#total mean
errors = [abs(userAverages[i]-test[find(x->(x==i), test[:,1]), 3]) for i=1:943]
err = zeros(1)
for e in errors  append!(err, e) end
print("MAE = ", mean(err[2:end]))


#Question 4
itemAverages = [mean(training[find(x->(x==i), training[:,2]), 3]) for i=1:1682]
itemAverages[find(x->(isnan(x)), itemAverages)]=globalMean
errors = [abs(itemAverages[i]-test[find(x->(x==i), test[:,2]), 3]) for i=1:1682]
err = zeros(1)
for e in errors  append!(err, e) end
print("MAE = ", mean(err[2:end]))


#Question 5
#array = readdlm("u.item", '|')
itemAverages = [mean(training[find(x->(x==i), training[:,2]), 3]) for i=1:1682]
#media dos itens desconsiderando os NaN
globalMeanItem = mean(itemAverages[find(x->(!isnan(x)), itemAverages)])

itemAverages[find(x->(isnan(x)), itemAverages)]=globalMeanItem
errors = [abs(itemAverages[i]-test[find(x->(x==i), test[:,2]), 3]) for i=1:1682]
err = zeros(1)
for e in errors  append!(err, e) end
print("MAE = ", mean(err[2:end]))
