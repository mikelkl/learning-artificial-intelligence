"""
Plotting tree nodes with text annotations
"""

import matplotlib.pyplot as plt

# Define box and arrow formatting
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    """
    Draws annotations with arrows.
    
    :param nodeTxt: node text
    :param centerPt: center point
    :param parentPt: parent point
    :param nodeType: node type
    :return: null
    """
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,
                            xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va='center', ha="center",
                            bbox=nodeType,
                            arrowprops=arrow_args)


def createPlot():
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    createPlot.ax1 = plt.subplot(111, frameon=False)
    plotNode('a decision node', (0.5, 0.1), (0.1, 0.5), decisionNode)
    plotNode('a leaf node', (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()


def getNumLeafs(myTree):
    """
    Traverse the entire tree and counts only the leaf nodes.
    
    :param myTree: dict-form tree
    :return: number of leaf nodes
    """
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    # firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():

        # Test if node is dictionary
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1

    return numLeafs


def getTreeDepth(myTree):
    """
    Count the number of times you hit a decision node.
    
    :param myTree: dict-form tree
    :return: depth of entire tree
    """
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    # firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth


def retrieveTree(i):
    listOfTrees = [{'no surfacing': {0: 'no', 1: {'flippers': \
                                                      {0: 'no', 1: 'yes'}}}},
                   {'no surfacing': {0: 'no', 1: {'flippers': \
                                                      {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                   ]
    return listOfTrees[i]


def plotMidText(cntrPt, parentPt, txtString):
    """
    Plots text between child and parent
    
    :param cntrPt: center point 
    :param parentPt: parent point
    :param txtString: text string
    :return: null
    """
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString)


def plotTree(myTree, parentPt, nodeTxt):
    # Get the width and height
    numLeafs = getNumLeafs(myTree)
    getTreeDepth(myTree)

    firstStr = list(myTree.keys())[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW,
              plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)  # Plot child value
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]

    # Decrement Y offset
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff),
                     cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD


def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()
