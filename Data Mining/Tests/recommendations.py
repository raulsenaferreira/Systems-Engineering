# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 08:41:33 2015

@author: raul
"""
from math import sqrt

critics={'Lisa Rose': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.5,
'Just My Luck': 3.0, 'Superman Returns': 3.5, 'You, Me and Dupree': 2.5,
'The Night Listener': 3.0},
'Gene Seymour': {'Lady in the Water': 3.0, 'Snakes on a Plane': 3.5,
'Just My Luck': 1.5, 'Superman Returns': 5.0, 'The Night Listener': 3.0,
'You, Me and Dupree': 3.5},
'Michael Phillips': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.0,
'Superman Returns': 3.5, 'The Night Listener': 4.0},
'Claudia Puig': {'Snakes on a Plane': 3.5, 'Just My Luck': 3.0,
'The Night Listener': 4.5, 'Superman Returns': 4.0,
'You, Me and Dupree': 2.5},
'Mick LaSalle': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0,
'Just My Luck': 2.0, 'Superman Returns': 3.0, 'The Night Listener': 3.0,
'You, Me and Dupree': 2.0},
'Jack Matthews': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0,
'The Night Listener': 3.0, 'Superman Returns': 5.0, 'You, Me and Dupree': 3.5},
'Toby': {'Snakes on a Plane':4.5,'You, Me and Dupree':1.0,'Superman Returns':4.0}}

#Euclidean distance metric
def sim_distance(prefs, person1, person2):
    si={}
    for item in prefs[person1]:
        if(item in prefs[person2]):
            si[item]=1
            
    if(len(si)==0): return 0
        
    sum_of_squares = sum([pow(prefs[person1][item] - prefs[person2][item], 2) 
    for item in prefs[person1] if item in prefs[person2]])

    return 1/sqrt(1+sum_of_squares)

#Pearson distance metric
def sim_pearson(prefs, p1, p2):
    si = {}
    
    for item in prefs[p1]:
        if item in prefs[p2]:
            si[item]=1
    
    if len(si)==0: return 0
    
    n = len(si)
    
    sum1 = sum([prefs[p1][it] for it in si])
    sum2 = sum([prefs[p2][it] for it in si])
    
    sumSq1 = sum([pow(prefs[p1][it], 2) for it in si])
    sumSq2 = sum([pow(prefs[p2][it], 2) for it in si])

    pSum = sum([prefs[p1][it] * prefs[p2][it] for it in si])
    
    num = pSum - (sum1*sum2/n)
    den = sqrt((sumSq1-pow(sum1, 2) / n) * (sumSq2 - pow(sum2, 2) / n))
    
    if den==0: return 0
    
    return num/den

#top 5 more similar to a given person
def topMatches(prefs, person, n=5, similarity = sim_pearson):
    scores = [(similarity(prefs, person, other), other) 
    for other in prefs if other!=person]
    
    scores.sort()
    scores.reverse()
    
    return scores[0:n]


# Gets recommendations for a person by using a weighted average
# of every other user's rankings
def getRecommendations(prefs,person,similarity=sim_pearson):
    totals={}
    simSums={}
    for other in prefs:
        # don't compare me to myself
        if other==person: continue
        sim=similarity(prefs,person,other)
        # ignore scores of zero or lower
        if sim<=0: continue
        for item in prefs[other]:
            # only score movies I haven't seen yet
            if item not in prefs[person] or prefs[person][item]==0:
                # Similarity * Score
                totals.setdefault(item,0)
                totals[item]+=prefs[other][item]*sim
                # Sum of similarities
                simSums.setdefault(item,0)
                simSums[item]+=sim
    # Create the normalized list
    rankings=[(total/simSums[item],item) for item,total in totals.items( )]
    # Return the sorted list
    rankings.sort( )
    rankings.reverse( )
    return rankings


def transformPrefs(prefs):
    result = {}
    for person in prefs:
        for item in prefs[person]:
            result.setdefault(item, {})
            #flip
            result[item][person] = prefs[person][item]
            
    return result
    
    
'''    
print sim_distance(critics, 'Lisa Rose', 'Gene Seymour')
print sim_person(critics, 'Lisa Rose', 'Gene Seymour')
'''
#print topMatches(critics, 'Toby', n=5, similarity = sim_distance)
#print topMatches(critics, 'Toby', n=5, similarity = sim_pearson)
#print getRecommendations(critics, 'Toby')
#print getRecommendations(critics, 'Toby', similarity = sim_distance)

print topMatches(transformPrefs(critics), 'Superman Returns')