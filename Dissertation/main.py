def main():
    #os.chdir('..')
    #\\Convidado\\Documents\\GitHub\\Systems-Engineering\\Dissertation
    path = os.getcwd()+'\\datasets\\'
    '''
    Reading NOAA dataset:
    Eight  features  (average temperature, minimum temperature, maximum temperature, dew
    point,  sea  level  pressure,  visibility,  average wind speed, maximum  wind  speed)
    are  used  to  determine  whether  each  day  experienced  rain  or no rain.
    '''
    dataValues = pd.read_csv(path+'noaa_data.csv',sep = ",")
    dataLabels = pd.read_csv(path+'noaa_label.csv',sep = ",")
    #Test sets: Predicting 365 instances by step. 50 steps. Starting labeled data with 5% of 365 instances.
    ''' 
    Test 0: 
    Two classes.
    K-Means + GMM / KDE
    '''
    test0(dataValues, dataLabels, 'gmm', excludingPercentage = 0.3)
    '''
    Test 1:
    Two classes.
    K-Means / SVM
    '''
    test1(dataValues, dataLabels, 'kmeans')