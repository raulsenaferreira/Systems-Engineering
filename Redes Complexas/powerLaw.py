from __future__ import division, absolute_import, print_function
from pylab import *  # for plotting
from numpy.random import *  # for random sampling
from graph_tool.all import *
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import sys
#import powerlaw

if sys.version_info < (3,):
    range = xrange

ccdf_patch = mpatches.Patch(color='green', label='CCDF')

########################## Dolphins
ug = Graph(directed=False)
#ug = GraphView(ug, vfilt=lambda v: v.out_degree() > 4)
ug = load_graph("/home/raul/workspace/Systems-Engineering/Redes Complexas/dolphins.gml")
d = ug.degree_property_map("out")
in_hist = vertex_hist(ug, d)

y = in_hist[0]
err = sqrt(in_hist[0])
err[err >= y] = y[err >= y] - 1e-2

a = sorted(in_hist[0][1:], reverse=True)
b = sorted(in_hist[1][1:-1], reverse=False)

figure(figsize=(6,4))
errorbar(b, a, fmt="o")
gca().set_yscale("log")
gca().set_xscale("log")
gca().set_ylim(0, 1e2)
gca().set_xlim(0, 1e2)
subplots_adjust(left=0.2, bottom=0.2)
xlabel("$k_{out}$")
ylabel("$NP(k_{out})$")
tight_layout()
#savefig("dolhin_degree_distribution.png")
values, base = np.histogram(a, bins=max(b))
cumulative = np.cumsum(values)

plt.legend(handles=[ccdf_patch])
plt.plot(base[:-1], len(a)-cumulative, c='green')
#plt.show()
savefig("dolphin_degree_distribution_and_ccdf.png")



########################## Politics books
ug = Graph(directed=False)
#ug = GraphView(ug, vfilt=lambda v: v.out_degree() > 4)
ug = load_graph("/home/raul/workspace/Systems-Engineering/Redes Complexas/politics.gml")
d = ug.degree_property_map("out")
in_hist = vertex_hist(ug, d)

y = in_hist[0]
err = sqrt(in_hist[0])
err[err >= y] = y[err >= y] - 1e-2

a = sorted(in_hist[0][1:], reverse=True)
b = sorted(in_hist[1][1:-1], reverse=False)

figure(figsize=(6,4))
errorbar(b, a, fmt="o")
gca().set_yscale("log")
gca().set_xscale("log")
gca().set_ylim(0.8, 1e2)
gca().set_xlim(0.8, 1e2)
subplots_adjust(left=0.2, bottom=0.2)
xlabel("$k_{out}$")
ylabel("$NP(k_{out})$")
tight_layout()
#savefig("politics_books_degree_distribution.png")
values, base = np.histogram(a, max(b))
cumulative = np.cumsum(values)

plt.legend(handles=[ccdf_patch])
plt.plot(base[:-1], len(a)-cumulative, c='green')
#plt.show()
savefig("politics_books_degree_distribution_and_ccdf.png")



########################### Political blogs
ug = Graph(directed=False)
#ug = GraphView(ug, vfilt=lambda v: v.out_degree() > 4)
ug = load_graph("/home/raul/workspace/Systems-Engineering/Redes Complexas/politicalBlogs.gml")
d = ug.degree_property_map("out")
in_hist = vertex_hist(ug, d)

y = in_hist[0]
err = sqrt(in_hist[0])
err[err >= y] = y[err >= y] - 1e-3

a = sorted(in_hist[0][1:], reverse=True)
b = sorted(in_hist[1][1:-1], reverse=False)

figure(figsize=(6,4))
errorbar(b, a, fmt="o")
gca().set_yscale("log")
gca().set_xscale("log")
gca().set_ylim(0.8, 1e3)
gca().set_xlim(0.8, 1e3)
subplots_adjust(left=0.2, bottom=0.2)
xlabel("$k_{out}$")
ylabel("$NP(k_{out})$")
tight_layout()
#savefig("political_blogs_degree_distribution.png")
values, base = np.histogram(a, bins=max(b))
cdf = np.cumsum(values)

plt.legend(handles=[ccdf_patch])
plt.plot(base[:-1], len(a)-cdf, c='green')

#plt.show()
savefig("political_blogs_degree_distribution_and_ccdf.png")
