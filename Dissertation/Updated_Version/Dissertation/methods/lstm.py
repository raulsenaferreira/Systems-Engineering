from source import classifiers
from source import metrics
from source import util
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

def start(dataValues, dataLabels, **kwargs):
    initialLabeledDataPerc = kwargs["initialLabeledDataPerc"]
    usePCA = kwargs["usePCA"]
    batches = kwargs["batches"]
    sizeOfBatch = kwargs["sizeOfBatch"]
    K = kwargs["K_variation"]
    clfName = kwargs["clfName"]
    classes = kwargs["classes"]

    print("METHOD: Sliding LSTM as classifier (Long-short term memory)")

    arrAcc = []
    initialDataLength = 0
    finalDataLength = round((initialLabeledDataPerc)*sizeOfBatch)
    
    #Initial labeled data
    X, y = util.loadLabeledData(dataValues, dataLabels, initialDataLength, finalDataLength, usePCA)
    initialDataLength=finalDataLength
    finalDataLength=sizeOfBatch
    
    for t in range(batches):
        #print(t)
        #clf = classifiers.svmClassifier(X, y)
        Ut, yt = util.loadLabeledData(dataValues, dataLabels, initialDataLength, finalDataLength, usePCA)
        #predicted = clf.predict(Ut)
        model = Sequential()
        model.add(LSTM(neurons, batch_input_shape=(batch_size, X[:,0], X[:,1]), stateful=True))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        
        arrAcc.append(metrics.evaluate(yt, predicted))
        
        initialDataLength=finalDataLength
        finalDataLength+=sizeOfBatch
        X = Ut
        y = predicted
    
    return "LSTM", arrAcc, X, y