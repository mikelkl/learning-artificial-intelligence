from numpy import *
import operator
from os import listdir


def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(intX, dataSet, labels, k):
    """
    k-Nearest Neighbors algorithm.
    
    :param intX: input vector
    :param dataSet: full matrix of training examples
    :param labels: vector of labels
    :param k: number of nearest neighbors to use in the voting
    :return: result label
    """
    dataSetSize = dataSet.shape[0]  # get instances number of training examples

    # Distance calculation using the Euclidian distance
    diffMat = tile(intX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistance = sqDiffMat.sum(axis=1)
    distances = sqDistance ** 0.5

    sortedDistIndices = distances.argsort()  # get the indices that would sort an array
    classCount = {}

    # Voting with lowest k distances
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1

    # Sort dictionary
    sortedClassCount = sorted(classCount.items(),  # set-like object providing a view on classCount's items
                              key=operator.itemgetter(1),
                              # callable object that fetches the given item(s) from its operand.
                              reverse=True)
    return sortedClassCount[0][0]


def file2matrix(filename):
    """
    Parsing data from a text file.
    
    :param filename: filename
    :return: input data matrix & target label vector
    """
    fr = open(filename)
    numberOfLines = len(fr.readlines())  # Get number of lines in file
    returnMat = zeros((numberOfLines, 3))  # Create NumPy matrix to return
    classLabelVector = []
    fr = open(filename)
    index = 0

    # Parse line to a list
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(listFromLine[-1])  # # to .append a LABEL, not an int()
        index += 1

    return returnMat, classLabelVector


def autoNorm(dataSet):
    """
    Normalizing numeric values.
    
    :param dataSet: input dataSet
    :return: normalized data set, ranges between max value and min value, min value 
    """
    minVals = dataSet.min(0)  # take the minimums from the columns
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))  # Element-wise division
    return normDataSet, ranges, minVals


def datingClassTest():
    """
    Testing the classifier.
    """
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :],
                                     datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %s, the real answer is: %s"
              % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
    print("the total error rate is: %f" % (errorCount / float(numTestVecs)))


def classifyPerson():
    resultDict = {'didntLike': 'not at all', 'smallDoses': 'in small doses', 'largeDoses': 'in large doses'}
    percentTats = float(input(
        "percentage of time spent playing video games?"
    ))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - minVals) / ranges, normMat, datingLabels, 3)
    print("You will probably like this person: ",
          resultDict[classifierResult])


def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')  # Get contents of directory
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))

    # Process class num from filename
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %s, the real answer is: %s"
              % (classifierResult, classNumStr))
        if classifierResult != classNumStr: errorCount += 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount / float(mTest)))