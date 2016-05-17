from sklearn import svm
from sklearn import metrics
def calculate_result(actual,pred):
    precision = metrics.precision_score(actual,pred);
    recall = metrics.recall_score(actual,pred)
    print 'predict info:'
    print 'precision:{0:.3f}'.format(precision)
    print 'recall:{0:0.3f}'.format(recall);
    print 'f1-score:{0:.3f}'.format(metrics.f1_score(actual, pred));


print "========================test SVC=================="

def loadDataSet(fileName):
    dataMat = [];
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split(',')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat

def SVC():
    data,label = loadDataSet(trainSet)
    test_data, test_label = loadDataSet(testSet)

    svc = svm.SVC()
    svc.fit(data,label)
    pred = svc.predict(test_data)
    calculate_result(test_label,pred)


if __name__ == '__main__':
    trainSet = 'train40.txt'
    testSet = 'testSet.txt'

    SVC()