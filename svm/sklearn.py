# from sklearn import *
# from sklearn.metrics import metrics
# def calculate_result(actual,pred):
#     precision = metrics.precision_score(actual,pred);
#     recall = metrics.recall_score(actual,pred)
#     print 'predict info:'
#     print 'precision:{0:.3f}'.format(precision)
#     print 'recall:{0:0.3f}'.format(recall);
#     print 'f1-score:{0:.3f}'.format(metrics.f1_score(actual, pred));
#
#
# print "========================test SVC=================="
#
# def loadDataSet(fileName):
#     dataMat = [];
#     labelMat = []
#     fr = open(fileName)
#     for line in fr.readlines():
#         lineArr = line.strip().split(',')
#         dataMat.append([float(lineArr[0]), float(lineArr[1])])
#         labelMat.append(float(lineArr[2]))
#     return dataMat, labelMat
#
#
# data,label = loadDataSet()
#
# svc = SVC()
# svc.fit(data,label)

