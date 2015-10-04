# Question 1
M = [5 10 -5 29 11; 1 33 15 3 5; 8 22 12 1 50; 3 11 39 20 2]
#[j == indmax(M[i, :]) ? print(1) : print(0) for i=1:size(M)[1], j=1:size(M)[2]] #substituir print por atribuição ao indice linear da matriz
[M[i, find([1:size(M)[2]].!=indmax(M[i,:]))]=0 for i=1:size(M)[1]]
[M[indmax( M )]=1 for i=1:size(M)[2]]
println(M)


# Question 2
M = [5 10 -5 29; 1 33 15 3; 8 22 12 1; 3 11 39 20]
[M[indmax(M)] = 0 for i=1:3]
println(M)


# Question 3
using StatsBase
#usar zscore
M = ceil(1*rand(4,4))
M2 = vcat(M, zeros(1,4), flipud(-M))
println(std(M2[:,1]))
println(mean(M2[:,1]))


# Question 4
#Pkg.add("Gadfly")
#Pkg.add("Cairo")
using Gadfly
plot(x=rand(10), y=rand(10))
arr = (2 - (-2)) .* rand(500,1) + (-2)
sin(arr[])
cos(arr[])
csc(arr[])
sec(arr[])


# Question 5
arr = 5 .* randn(100,1) + 20
