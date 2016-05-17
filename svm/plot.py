from numpy import *
import matplotlib
import matplotlib.pyplot as plt
def loadDataSet(fileName):
    dataMat = [];
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split(',')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat



def showdata():
    dataMat, labelMat = loadDataSet('testSetRBF2.txt')
    matrix = mat(dataMat)
    #labels= mat(labelMat)
    plt.figure(figsize=(8,5),dpi=80)
    axes = plt.subplot(111)
    type1_x = []
    type1_y = []
    type2_x = []
    type2_y = []
    type3_x = []
    type3_y = []
    print range(len(labelMat))
    for i in range(len(labelMat)):
        if labelMat[i] == 1:
            type1_x.append(dataMat[i][0])
            type1_y.append(dataMat[i][1])
        if labelMat[i] == -1:
            type2_x.append(dataMat[i][0])
            type2_y.append(dataMat[i][1])
    type1 = axes.scatter(type1_x, type1_y, s=40, c='red')
    type2 = axes.scatter(type2_x, type2_y, s=40, c='green')
    plt.xlabel(u'1st')
    plt.ylabel(u'2nd')
    axes.legend((type1, type2), (u'pos', u'neg'), loc=2)

    plt.show()



if __name__ == '__main__':
    showdata()