# Questions in firstList.pdf
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
# Jeito 1 com matriz toda retornando desvio padrão 1 e média 0
M=zscore(M)
println(std(M))
println(mean(M))

# Jeito 2 com coluna retornando desvio padrão 1 e média 0
#M = ceil(1*rand(4,4))
#M2 = vcat(M, zeros(1,4), flipud(-M))
#println(std(M2[:,1]))
#println(mean(M2[:,1]))


# Question 4
#using PyPlot
using PyCall
@pyimport matplotlib.pyplot as plt

up = 2; low = -2
arr = (up - low) .* rand(500,1) + low
#arr = squeeze(reshape(arr, 1, 500), 1)
u = linspace(0, 2*pi, 100)

u_sinx = sin(u)
u_cosx = cos(u)
u_secx = sec(u)
u_cscx = csc(u)

transf = [u_sinx] * arr'
transf2 = [u_cosx] * arr'
transf3 = [u_secx] * arr'
transf4 = [u_cscx] * arr'

#plot
x=u

y=transf
plt.plot(x, y, color="red", linewidth=2.0, linestyle="--")
#plt.title("Seno")
#plt.show()

y=transf2
plt.plot(x, y, color="red", linewidth=2.0, linestyle="--")
#plt.title("Cosseno")
plt.title("Seno e cosseno")
plt.show()

y=transf2
plt.plot(x, y, color="red", linewidth=2.0, linestyle="--")
#plt.title("Secante")
#plt.show()

y=transf4
plt.plot(x, y, color="red", linewidth=2.0, linestyle="--")
#plt.title("Cossecante")
plt.title("Secante e cossecante")
plt.show()

#subplot
plt.subplot(121)
plt.plot(u_secx, u_cscx)
plt.axis(:equal)
plt.title("Secante")

plt.subplot(122)
plt.plot(transf3[:, 1], transf4[:, 1])
plt.axis(:equal)
plt.title("Cossecante")
plt.show()

plt.subplot(121)
plt.plot(u_sinx, u_cosx)
plt.axis(:equal)
plt.title("Seno")

plt.subplot(122)
plt.plot(transf[:, 1], transf2[:, 1])
plt.axis(:equal)
plt.title("Cosseno")
plt.show()


# Question 5
arr = 5 .* randn(100,1) + 20
u = linspace(0, 2*pi, 100)
plt.plot(u, arr, color="red", linewidth=2.0, linestyle="--")
plt.title("Distribuição")
plt.show()


# Question 6
