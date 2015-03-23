from collections import Counter
from pprint import pprint

def calcBayes(trainingSetFeatures, traininSetClasses, testSetFeatures):
	count_classes = Counter()

	for c in traininSetClasses:
		count_classes.update(c)
		
	prob_classes = {}
	prob_classes['S'] = count_classes['S']/ len(traininSetClasses)
	prob_classes['N'] = count_classes['N']/ len(traininSetClasses)
	
	count_features = []
	prob_features = []
	
	for i in xrange(0,4):
		count_features.append(Counter())
		prob_features.append({})
	
	for j in xrange(0,4):
		for i in xrange(0, len(traininSetClasses)):
			count_features[j].update( [(trainingSetFeatures[i][j], traininSetClasses[i])] )
		
		for ((f, c), counting) in count_features[j].items():
			prob_features[j][ (f, c) ] = count_features[j][ ( f, c) ] / count_classes[ c ]	
			
	prob_test_set = {}
	
	for i in xrange(len(testSetFeatures)):
		for j in xrange(0,4):
			for c in ['S', 'N']:
				if (testSetFeatures[i][j], c) in prob_features[j]:
					if str(testSetFeatures[i]) in prob_test_set:
						prob_test_set[ (str(testSetFeatures[i]), c) ] *= prob_features[j][ (testSetFeatures[i][j], c) ]
					else:
						prob_test_set[ (str(testSetFeatures[i]), c) ] = prob_features[j][ (testSetFeatures[i][j], c) ]
			
					prob_test_set[ (str(testSetFeatures[i]), c) ] *= prob_classes[c]
	
	test_classifications = []
	for i in xrange(len(testSetFeatures)):
		if prob_test_set[ (str(testSetFeatures[i]), 'S') ] >= prob_test_set[ (str(testSetFeatures[i]), 'N') ]:
			test_classifications.append('S')
		else:
			test_classifications.append('N')
			
	pprint(prob_test_set)
	return test_classifications					
		

def main():
	features = [[40, 'M', 'N', 'E'],
				[20, 'A', 'N', 'B'],
				[20, 'A', 'N', 'B'],
				[30, 'A', 'N', 'B'],
				[40, 'M', 'N', 'B'],
				[40, 'B', 'S', 'B'],
				[40, 'B', 'S', 'E'],
				[30, 'B', 'S', 'E'],
				[20, 'M', 'N', 'B'],
				[20, 'B', 'S', 'B'],
				[40, 'M', 'S', 'B'],
				[40, 'M', 'S', 'E'],
				[30, 'M', 'N', 'E'],
				[30, 'A', 'S', 'B'],
				[40, 'M', 'N', 'E'],
				[40, 'M', 'N', 'E']]
	
	classes = ['N' ,'N','N', 'S', 'S', 'S', 'N', 'S', 'N', 'S', 'S', 'S', 'S', 'S', 'N', 'N']	
	
	dois_tercos = 2 * (len(features) / 3)
	trainingSetFeatures = features[0:dois_tercos]
	traininSetClasses = classes[0:dois_tercos]
	
	testSetFeatures = features[dois_tercos:]
	testSetClasses = classes[dois_tercos:]
	
	pprint(testSetClasses)
	pprint(calcBayes(trainingSetFeatures, traininSetClasses, testSetFeatures))
	
main()
	

	
	
