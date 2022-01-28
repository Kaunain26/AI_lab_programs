class Graph:
    def __init__(self, graph, huristicNodeList, startNode):
        self.graph = graph
        self.H = huristicNodeList
        self.start = startNode
        self.parent = {}
        self.status = {}
        self.solutionGraph = {}

    def applyAOStar(self):
        self.aoStar(self.start, False)

    def getNeighbours(self, v):
        return self.graph.get(v, '')

    def getStatus(self, v):
        return self.status.get(v, 0)

    def setStatus(self, v, val):
        self.status[v] = val

    def getHeuristicNodeValue(self, n):
        return self.H.get(n, 0)

    def setHeuristicNodeValue(self, n, value):
        self.H[n] = value

    def printSolution(self):
        print("FOR GRAPH SOLUTION , TRAVERSE THE GRAPH FROM THE START NODE", self.start)
        print("..................................................")
        print(self.solutionGraph)
        print("..................................................")

    def computeMinimumCostChildNode(self, v):
        minimumCost = 0
        costToChildNodeListDict = {}
        costToChildNodeListDict[minimumCost] = []
        flag = True
        for nodeInfoTupleList in self.getNeighbours(v):
            # print("....nodeInfoTupleList " + str(nodeInfoTupleList))

            cost = 0
            nodeList = []
            for c, weight in nodeInfoTupleList:
                # print("...c, weight ", c, weight)

                cost = cost + self.getHeuristicNodeValue(c) + weight
                nodeList.append(c)
            # print("...nodeList and cost ", nodeList, cost)
            if flag == True:
                minimumCost = cost
                costToChildNodeListDict[minimumCost] = nodeList
                flag = False
                # print("...costToChildNodeListDict\n ", costToChildNodeListDict)
            else:
                if minimumCost > cost:
                    minimumCost = cost
                    costToChildNodeListDict[minimumCost] = nodeList

        return minimumCost, costToChildNodeListDict[minimumCost]

    def aoStar(self, v, backTracking):
        print("\nHEURISTIC VALUES :", self.H)
        print("SOLUTION GRAPH :", self.solutionGraph)
        print("PROCESSING NODE :", v)
        print("....................................................................")
        if self.getStatus(v) >= 0:
            # print("....getStatus " + str(self.getStatus(v)) + "")
            minimumCost, childNodeList = self.computeMinimumCostChildNode(v)

            print(minimumCost, childNodeList)
            self.setHeuristicNodeValue(v, minimumCost)
            self.setStatus(v, len(childNodeList))
            # print("len-childNodeList: " + str(len(childNodeList)))

            solved = True
            for childNode in childNodeList:
                # print("childNode from childNodeList: " + childNode)
                self.parent[childNode] = v
                # print("parentNode: ", self.parent)
                if self.getStatus(childNode) != 1:
                    solved = solved & False
            if solved == True:
                self.setStatus(v, -1)
                self.solutionGraph[v] = childNodeList
            if v != self.start:
                self.aoStar(self.parent[v], True)
            if backTracking == False:
                for childNode in childNodeList:
                    self.setStatus(childNode, 0)
                    self.aoStar(childNode, False)


print("Graph - 1")
h1 = {'A': 1, 'B': 6, 'C': 2, 'D': 12, 'E': 2, 'F': 1, 'G': 5, 'H': 7, 'I': 7, 'J': 1}
graph1 = {
    'A': [[('B', 1), ('C', 1)], [('D', 1)]],
    'B': [[('G', 1)], [('H', 1)]],
    'C': [[('J', 1)]],
    'D': [[('E', 1), ('F', 1)]],
    'G': [[('I', 1)]]
}

G1 = Graph(graph1, h1, 'A')
G1.applyAOStar()
G1.printSolution()

# output

# Graph - 1
#
# HEURISTIC VALUES : {'A': 1, 'B': 6, 'C': 2, 'D': 12, 'E': 2, 'F': 1, 'G': 5, 'H': 7, 'I': 7, 'J': 1}
# SOLUTION GRAPH : {}
# PROCESSING NODE : A
# ....................................................................
# 10 ['B', 'C']
#
# HEURISTIC VALUES : {'A': 10, 'B': 6, 'C': 2, 'D': 12, 'E': 2, 'F': 1, 'G': 5, 'H': 7, 'I': 7, 'J': 1}
# SOLUTION GRAPH : {}
# PROCESSING NODE : B
# ....................................................................
# 6 ['G']
#
# HEURISTIC VALUES : {'A': 10, 'B': 6, 'C': 2, 'D': 12, 'E': 2, 'F': 1, 'G': 5, 'H': 7, 'I': 7, 'J': 1}
# SOLUTION GRAPH : {}
# PROCESSING NODE : A
# ....................................................................
# 10 ['B', 'C']
#
# HEURISTIC VALUES : {'A': 10, 'B': 6, 'C': 2, 'D': 12, 'E': 2, 'F': 1, 'G': 5, 'H': 7, 'I': 7, 'J': 1}
# SOLUTION GRAPH : {}
# PROCESSING NODE : G
# ....................................................................
# 8 ['I']
#
# HEURISTIC VALUES : {'A': 10, 'B': 6, 'C': 2, 'D': 12, 'E': 2, 'F': 1, 'G': 8, 'H': 7, 'I': 7, 'J': 1}
# SOLUTION GRAPH : {}
# PROCESSING NODE : B
# ....................................................................
# 8 ['H']
#
# HEURISTIC VALUES : {'A': 10, 'B': 8, 'C': 2, 'D': 12, 'E': 2, 'F': 1, 'G': 8, 'H': 7, 'I': 7, 'J': 1}
# SOLUTION GRAPH : {}
# PROCESSING NODE : A
# ....................................................................
# 12 ['B', 'C']
#
# HEURISTIC VALUES : {'A': 12, 'B': 8, 'C': 2, 'D': 12, 'E': 2, 'F': 1, 'G': 8, 'H': 7, 'I': 7, 'J': 1}
# SOLUTION GRAPH : {}
# PROCESSING NODE : I
# ....................................................................
# 0 []
#
# HEURISTIC VALUES : {'A': 12, 'B': 8, 'C': 2, 'D': 12, 'E': 2, 'F': 1, 'G': 8, 'H': 7, 'I': 0, 'J': 1}
# SOLUTION GRAPH : {'I': []}
# PROCESSING NODE : G
# ....................................................................
# 1 ['I']
#
# HEURISTIC VALUES : {'A': 12, 'B': 8, 'C': 2, 'D': 12, 'E': 2, 'F': 1, 'G': 1, 'H': 7, 'I': 0, 'J': 1}
# SOLUTION GRAPH : {'I': []}
# PROCESSING NODE : B
# ....................................................................
# 2 ['G']
#
# HEURISTIC VALUES : {'A': 12, 'B': 2, 'C': 2, 'D': 12, 'E': 2, 'F': 1, 'G': 1, 'H': 7, 'I': 0, 'J': 1}
# SOLUTION GRAPH : {'I': [], 'B': ['G']}
# PROCESSING NODE : A
# ....................................................................
# 6 ['B', 'C']
#
# HEURISTIC VALUES : {'A': 6, 'B': 2, 'C': 2, 'D': 12, 'E': 2, 'F': 1, 'G': 1, 'H': 7, 'I': 0, 'J': 1}
# SOLUTION GRAPH : {'I': [], 'B': ['G']}
# PROCESSING NODE : C
# ....................................................................
# 2 ['J']
#
# HEURISTIC VALUES : {'A': 6, 'B': 2, 'C': 2, 'D': 12, 'E': 2, 'F': 1, 'G': 1, 'H': 7, 'I': 0, 'J': 1}
# SOLUTION GRAPH : {'I': [], 'B': ['G']}
# PROCESSING NODE : A
# ....................................................................
# 6 ['B', 'C']
#
# HEURISTIC VALUES : {'A': 6, 'B': 2, 'C': 2, 'D': 12, 'E': 2, 'F': 1, 'G': 1, 'H': 7, 'I': 0, 'J': 1}
# SOLUTION GRAPH : {'I': [], 'B': ['G']}
# PROCESSING NODE : J
# ....................................................................
# 0 []
#
# HEURISTIC VALUES : {'A': 6, 'B': 2, 'C': 2, 'D': 12, 'E': 2, 'F': 1, 'G': 1, 'H': 7, 'I': 0, 'J': 0}
# SOLUTION GRAPH : {'I': [], 'B': ['G'], 'J': []}
# PROCESSING NODE : C
# ....................................................................
# 1 ['J']
#
# HEURISTIC VALUES : {'A': 6, 'B': 2, 'C': 1, 'D': 12, 'E': 2, 'F': 1, 'G': 1, 'H': 7, 'I': 0, 'J': 0}
# SOLUTION GRAPH : {'I': [], 'B': ['G'], 'J': []}
# PROCESSING NODE : A
# ....................................................................
# 5 ['B', 'C']
# FOR GRAPH SOLUTION , TRAVERSE THE GRAPH FROM THE START NODE A
# ..................................................
# {'I': [], 'B': ['G'], 'J': []}
# ..................................................#
