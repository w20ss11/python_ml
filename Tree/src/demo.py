'''
Created on 

@author: ly
'''
from imp import reload
from src import trees
from src import treePlotter
from src.treePlotter import getNumLeafs
reload(trees)
myDat,labels=trees.createDataSet()


myTree=trees.createTree(myDat, labels)
print(myTree)

print('num leafs',getNumLeafs(myTree))

myTree=treePlotter.retrieveTree(0)
treePlotter.createPlot(myTree)