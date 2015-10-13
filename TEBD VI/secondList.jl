using Gadfly

#Question 1
array = readdlm("path/u.data")
qtyPerUser = hcat(hist(array[:,1], 0.1:943.9)[2], 1:943)
qtyPerUser = sortrows(qtyPerUser,by=x->(x[1]), rev=true)
plot(x=qtyPerUser[:,2], y=qtyPerUser[:,1], Geom.line)

#Question 2
plot(x=1:5, y=hist(array[:,3], 0:5.9)[2], Geom.line)
