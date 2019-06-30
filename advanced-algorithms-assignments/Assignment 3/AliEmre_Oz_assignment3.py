__author__ = "aliemre"
#Ali Emre Oz
#213950785


#Question 1
def bestRoute(mountain):

    for i in range(1, len(mountain)):
        for j in range(len(mountain[i])):
            if j == 0:
                mountain[i][j] = mountain[i][j] + mountain[i - 1][j]
            elif j == len(mountain[i - 1]):
                mountain[i][j] = mountain[i][j] + mountain[i - 1][j - 1]
            else:
                mountain[i][j] = max(mountain[i][j] + mountain[i - 1][j], mountain[i][j] + mountain[i - 1][j - 1])


    maxScore = max(mountain[len(mountain)-1])
    maxIndex = len(mountain)-1,mountain[len(mountain)-1].index(maxScore)
    path = []
    path.append(maxIndex)

    def findUp(mountain,index):
        upRow = index[0]
        upCol = index[1]
        subList = [mountain[upRow-1][upCol-1],mountain[upRow-1][upCol]]
        upMax = max(subList)
        return upMax,(upRow-1,mountain[upRow-1].index(upMax))

    i = len(mountain) - 2
    index = maxIndex
    while i > 0:
        path.append(findUp(mountain,index)[1])
        index = findUp(mountain,index)[1]
        i = i - 1
    path.append((0,0))
    path.reverse()
    return "Route: " + str(path) + "\n" + "Score: " + str(maxScore)


#Question 2
class Conference:
    def __init__(self, id, start, end, participant):
        self.id = id
        self.start = start
        self.end = end
        self.participant = participant
def findMaxParticipant(conference):
    n = len(conference)
    conference = sorted(conference, key=lambda j: j.end)
    conference_matrices = []
    for i in range(n):
        conference_matrices.append(0)

    conference_matrices[0] = conference[0].participant
    i = 1
    while i<n:
        p = conference[i].participant
        c = checkConflictOverConference(conference, i)
        if (c != False):
            p = p + conference_matrices[c]
        conference_matrices[i] = max(p, conference_matrices[i - 1])
        i = i + 1
    total_participant = conference_matrices[-1]
    return total_participant
def checkConflictOverConference(conference,i):
    j = i - 1
    while j >= 0:
        if conference[j].end <= conference[i].start:
            return j
        j = j - 1
    return False
def bestSelection(conferences):
    conferencesList = []
    for i,j in conferences.items():
        id = i.split(" ")[1]
        start = j[0]
        end = j[1]
        participant = j[2]
        a = Conference(id, start, end, participant)
        conferencesList.append(a)
    return "Total number of	participants: "+str(findMaxParticipant(conferencesList))

#Question 3
allList = []
def createList(p,m):
    global allList
    subList = []
    x = 0
    while x<m:
        subList.append(p[x])
        x = x + 1
    allList.append(subList)
def partition(number):
    pList = []
    pointer = 0
    pList.insert(pointer, number)
    while True:
        createList(pList,pointer+1)
        a = 0
        while (pointer>=0) and (pList[pointer]==1):
            a = a + pList[pointer]
            pointer = pointer - 1
        if pointer<0:
            return
        pList[pointer] = pList[pointer] - 1
        a = a + 1
        while a>pList[pointer]:
            pList[pointer+1] = pList[pointer]
            a = a - pList[pointer]
            pointer = pointer + 1
        pList.insert(pointer+1,a)
        pointer = pointer + 1
def possibleCombinations(n):
    partition(n)
    allList.remove([n])
    allOperation = []
    for i in allList:
        operation = ""
        for j in i:
            operation = operation+str(j)+"+"
        allOperation.append(operation[:-1])
    for i in allOperation:
        print i
    return "Done"



#TEST
#print "Question 1"
#mountain=[[1],[1,2],[3,5,4],[2,6,7,4]]
#print bestRoute(mountain)
#print "*"*50
#print "Question 2"
#conferences=	{"Conference 1":	[1300,	1559,	300],"Conference 2":[1100,  1359,	500],"Conference 3":[1600,	1759,	200]}
#print bestSelection((conferences))
#print "*"*50
#print "Question 3"
#print possibleCombinations(5)