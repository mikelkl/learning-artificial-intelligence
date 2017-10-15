from math import log
import operator


def calcShannonEnt(dataSet):
    """
    Calculate the Shannon entropy of a dataset.
    :param dataSet: input dataset 
    :return: Shannon entropy
    """
    numEntries = len(dataSet)
    labelCounts = {}

    # Create dictionary of all possible classes
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        # Logarithm base entropy formula
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


def splitDataSet(dataSet, axis, value):
    """
    Split dataset on a given feature.
    
    :param dataSet: dataset to be split
    :param axis: feature to be split on
    :param value: value of the feature to return
    :return: separate list
    """
    retDatSet = []  # Create separate list
    for featVec in dataSet:
        if featVec[axis] == value:
            # Cut out the feature split on
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDatSet.append(reducedFeatVec)
    return retDatSet


def chooseBestFeatureToSplit(dataSet):
    """
    Choosing the best feature to split on.
    
    :param dataSet: input data set
    :return: best feature index
    """
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        # Create unique list of class labels
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)

        newEntropy = 0.0

        # Calculate entropy for each split
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            # Find the best information gain
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def majorityCnt(classList):
    """
    Take majority vote for the class of leaf node.
    
    :param classList: input class list
    :return: class that occurs with the greatest frequency
    """
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    """
    Building tree.
    
    :param dataSet: input dataset 
    :param labels: list of labels contains a label for each of the features in the dataset 
    :return: built tree
    """
    classList = [example[-1] for example in dataSet]  # create a list of all the class labels in input dataset

    # Stop when all classes are equal
    if classList.count(classList[0]) == len(classList):
        return classList[0]

    # When no more features, return majority
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)

    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}

    # Get list of unique values
    del labels[bestFeat]
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)

    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


def classify(inputTree, featLabels, testVec):
    """
    Classification for an existing decision tree
    
    :param inputTree: input tree
    :param featLabels: feature labels
    :param testVec: test vector
    :return: class label
    """
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)  # Translate label string to index
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'wb')
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(filename):
    import pickle
    fr = open(filename,'rb')
    return pickle.load(fr)
