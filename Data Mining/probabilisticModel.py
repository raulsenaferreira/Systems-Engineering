# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 15:55:08 2015

@author: raul
"""

from numpy.random import choice

#globals
genreType = ['action', 'romance']
occupationType = ['student', 'homemaker', 'engineer']
genderType = ['F', 'M']
ageNumber = range(20, 41)
ratingsNumber = [1,2,3,4,5]
    
def main():
    populationSize = 500
    #choose random
    genres=generateValuesByProbability(genreType, [0.55, 0.45], populationSize)
    occupations=generateValuesByProbability(occupationType, [0.5, 0.25, 0.25], populationSize)
    genders=generateValuesByProbability(genderType, [0.6, 0.4], populationSize)
    #ages=generateValuesByProbability(ageNumber, range(5, ), populationSize)#50% each
    
    generateRating(genres, occupations, genders)

    
def generateValuesByProbability(collection, weights, size):
    population = []
    for i in range(0, size):
        population.append(choice(collection, p=weights))
    return population


def generateRating(genres, occupations, genders):
    ratings=[]
    combinations=[genreType[0]+occupationType[0]+genderType[0],
                  genreType[1]+occupationType[0]+genderType[0]]
    
    for i, j, k in zip(genres, occupations, genders):
        combination = i+j+k
        
        if (combination == combinations[0]):
            ratings.append(generateValuesByProbability(ratingsNumber, [0.1, 0.4, 0.3, 0.1, 0.1], 1))
        elif (combination == combinations[1]):
            ratings.append(generateValuesByProbability(ratingsNumber, [0.1, 0.1, 0.2, 0.4, 0.2], 1))

main()