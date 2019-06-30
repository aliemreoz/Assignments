#Ali Emre Oz
#213950785

__author__ = "aliemre"
import heapq
class PriorityQueue:
    def __init__(self, inlist=None):
        self.priorityqueue = inlist or []
        heapq.heapify(self.priorityqueue)

    def makequeue(self, inlist):
        self.priorityqueue = inlist
        heapq.heapify(self.priorityqueue)

    def insert(self, item):
        heapq.heappush(self.priorityqueue, item)

    def peek(self):
        if len(self.priorityqueue) > 0:
            return self.priorityqueue[0]
        else:
            return None

    def isEmpty(self):
        if len(self.priorityqueue) == 0:
            return True
        else:
            return False

    def deleteMin(self):
        if len(self.priorityqueue) > 0:
            return heapq.heappop(self.priorityqueue)
        else:
            return None

    def decreaseKey(self, item):
        # heapq does not have method to update keys
        # Here, searching the priority queue list to find the item (specifically second value in tuple item. needs to be unique),
        # then updating the item and re-heapifying
        # VERY ineffecient way to implement this function. If this operation is used a lot, consider implementing your own heap
        for i in range(len(self.priorityqueue)):
            if self.priorityqueue[i][1] == item[1]:
                self.priorityqueue[i] = item
                break
        heapq.heapify(self.priorityqueue)
neighbours = {}
cost = {}
def getNeighboursDict(G,state):
    m = len(G)
    n = len(G[0])
    x = state[0]
    y = state[1]
    if (x == 0) and (y == 0):
        neighbours[state] = [(x+1,y),(x,y+1)]
    elif (x == m-1) and (y == 0):
        neighbours[state] = [(x-1,y),(x,y+1)]
    elif (x == 0) and (y == n-1):
        neighbours[state] = [(x,y-1),(x+1,y)]
    elif (x == m-1) and (y == n-1):
        neighbours[state] = [(x,y-1),(x-1,y)]
    elif (0 < x < m-1) and (y == 0):
        neighbours[state] = [(x-1,y),(x,y+1),(x+1,y)]
    elif (x == 0) and (0<y<n-1):
        neighbours[state] = [(x,y+1), (x,y-1), (x+1,y)]
    elif (x == m-1) and (0<y<n-1):
        neighbours[state] = [(x,y+1), (x,y-1), (x-1,y)]
    elif (0 < x < m-1) and (y == n-1):
        neighbours[state] = [(x-1,y),(x+1,y),(x,y-1)]
    else:
        neighbours[state] = [(x-1, y), (x+1, y), (x, y-1), (x,y+1)]
    return neighbours
def createNeighbourDict(G):
    m = len(G)
    n = len(G[0])
    for i in range(m):
        for j in range(n):
            getNeighboursDict(G,(i,j))
    return neighbours
def createCostDict(G):
    m = len(G)
    n = len(G[0])
    for i in range(m):
        for j in range(n):
            cost[(i,j)]=G[i][j]
    return cost
def jump(start,jump_point,main_cost):
    cost = 9999
    next_point = (None,None)
    try:
        start_x = start[0]
        start_y = start[1]
        jump_point_x = jump_point[0]
        jump_point_y = jump_point[1]
        if (start_x == jump_point_x) and (start_y<jump_point_y):
            cost = main_cost[(jump_point_x,jump_point_y+1)]*2
            next_point = (jump_point_x,jump_point_y+1)

        elif (start_x == jump_point_x) and (start_y>jump_point_y):
            cost = main_cost[(jump_point_x,jump_point_y-1)]*2
            next_point = (jump_point_x,jump_point_y-1)

        elif (start_y == jump_point_y) and (start_x<jump_point_x):
            cost = main_cost[(jump_point_x+1,jump_point_y)]*2
            next_point = (jump_point_x+1,jump_point_y)

        elif (start_y == jump_point_y) and (start_x>jump_point_x):
            cost = main_cost[(jump_point_x-1,jump_point_y)]*2
            next_point = (jump_point_x-1,jump_point_y)

    except: KeyError
    return cost,next_point
start_point = (2,2)
path = []
path.append(start_point)
def next_move(start_point,main_neighbour,main_cost):

    cost_of_neighbours = {}
    for i in main_neighbour[start_point]:
        cost_of_neighbours[i] = main_cost[i]

    for i,j in cost_of_neighbours.items():
        if j == "X":
            new_p = jump(start_point,i,main_cost)[1]
            cost_of_neighbours.pop(i)
            cost_of_neighbours[new_p] = jump(start_point,i,main_cost)[0]
    return cost_of_neighbours
def get_neighbour_cost(points,cost,main_neighbour,main_cost):
    points_x = points[0]
    points_y = points[1]
    next_moves = next_move((points_x,points_y),main_neighbour,main_cost)
    neighbours_cost = []
    for i,j in next_moves.items():
        if j != 9999:
            neighbours_cost.append(((j+cost),i))
    return neighbours_cost
def find_shortest_path(G):
    main_cost = createCostDict(G)
    main_neighbour = createNeighbourDict(G)
    finish_x = len(G) - 1
    finish_y = len(G[0]) - 1
    finish = (finish_x,finish_y)
    H = PriorityQueue()
    best_way = {(0,0): None}
    on = (0, (0,0))
    to = on[1]
    cost = on[0]

    ###
    while to != finish:
        points = get_neighbour_cost(to, cost,main_neighbour,main_cost)
        while len(points)>0:
            for point in points:
                if point[1] not in best_way:
                    H.insert(point)
                    best_way[point[1]] = to
                points.remove(point)

        on = H.deleteMin()
        to = on[1]
        cost = on[0]
    ###


    ###
    step = best_way[on[1]]
    path = []
    bool = 1
    while (bool == 1):
        if step != None:
            path.append(step)
            step = best_way[step]
            bool = 1
        else:
            bool = 0
    ###

    main_path = path[::-1]
    main_path.append(finish)
    total_cost = on[0]
    total_step = len(main_path) - 1
    path_as_str = ""
    for i in main_path:
        path_as_str = path_as_str + str(i) + ", "


    print "Minimum cost : "+str(total_cost)
    print "Steps : "+str(total_step)
    print "Path : "+path_as_str[:-2]


#find_shortest_path([[0, "X", 1, 4, 9, "X"], [7, 7, 4, "X", 4, 8], [3, "X", 3, 2, "X", 4], [10, 2, 5, "X", 3, 0]])


