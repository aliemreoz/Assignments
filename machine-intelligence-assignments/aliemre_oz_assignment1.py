#initial features of spesific 8-Puzzle  positions.
#initially it has no parents and it's level is equal to 0.
initial_state = [[5, 4, 0], [6,1,8], [7,3,2]]
initial_parent = None
initial_level = 0


#it creates 1D array based on preferred 2D array. Ex: [[5,4,0],[6,1,8],[7,3,2]] to [5,4,0,6,1,8,7,3,2]
#it takes 2D array and divide each subarrays to another arrays and then merge that arrays on final array.
def break2Darray(initial_state):
    list1 = initial_state[0]
    list2 = initial_state[1]
    list3 = initial_state[2]
    initial_state_1D = list1 + list2 + list3
    return initial_state_1D


#up movement for 8-Puzzle
#it looks for 0 and check it's position. Based on it's position it swap 0's and possible number's position.
def up(state):
    child_state = state[:]
    idx = child_state.index(0)
    if idx not in [0, 1, 2]:
        temp = child_state[idx-3]
        child_state[idx-3] = child_state[idx]
        child_state[idx] = temp
        return child_state
    else:
        return None


#down movement for 8-Puzzle
#it looks for 0 and check it's position. Based on it's position it swap 0's and possible number's position.
def down(state):
    child_state = state[:]
    idx = child_state.index(0)
    if idx not in [6, 7, 8]:
        temp = child_state[idx+3]
        child_state[idx+3] = child_state[idx]
        child_state[idx] = temp
        return child_state
    else:
        return None


#left movement for 8-Puzzle
#it looks for 0 and check it's position. Based on it's position it swap 0's and possible number's position.
def left(state):
    child_state = state[:]
    idx = child_state.index(0)
    if idx not in [0, 3, 6]:
        temp = child_state[idx-1]
        child_state[idx-1] = child_state[idx]
        child_state[idx] = temp
        return child_state
    else:
        return None


#right movement for 8-Puzzle
#it looks for 0 and check it's position. Based on it's position it swap 0's and possible number's position.
def right(state):
    child_state = state[:]
    idx = child_state.index(0)
    if idx not in [2, 5, 8]:
        temp = child_state[idx+1]
        child_state[idx+1] = child_state[idx]
        child_state[idx] = temp
        return child_state
    else:
        return None


#try to create childss of given state.
#it try to do previously defined movements. If it can't to it returns None. At the end it deletes the None(unpossible movements).
def createChilds(state):
    state = break2Darray(state)
    childs = []
    childs.append(right(state))
    childs.append(left(state))
    childs.append(up(state))
    childs.append(down(state))
    for j in childs:
        if j == None:
            childs.remove(j)
    for i in childs:
        if i == None:
            childs.remove(i)
    return childs



#One of the main functions of assingment.
#It does the createChilds function and convert the result to 2D array and append it on final list.
def expand(initial_state):
    expanded_states = []
    for state in createChilds((initial_state)):
        states = []
        states.append([])
        states.append([])
        states.append([])
        states[0].append(state[0])
        states[0].append(state[1])
        states[0].append(state[2])
        states[1].append(state[3])
        states[1].append(state[4])
        states[1].append(state[5])
        states[2].append(state[6])
        states[2].append(state[7])
        states[2].append(state[8])
        expanded_states.append(states)
    return expanded_states


# creating object for possible childs
class Child:
    def __init__( self, state, parent, level):
        self.state = state
        self.parent = parent
        self.level = level


#function for creating child object
def create(state,parent,level):
    return Child(state,parent,level)


tempSolutions = []
possibleSolutions = []


#try to create child for initial states.
def firstCheck(initial):
    tempSolutions.append(create(up(initial.state), initial,  initial.level + 1))
    tempSolutions.append(create(down(initial.state), initial, initial.level + 1))
    tempSolutions.append(create(left(initial.state), initial, initial.level + 1))
    tempSolutions.append(create(right(initial.state), initial, initial.level + 1))
    for j in tempSolutions:
        if j.state == None:
            tempSolutions.remove(j)
    for i in tempSolutions:
        if i.state == None:
            tempSolutions.remove(i)
    return tempSolutions


#try to create child for second time.
def secondCheck(initial):
    for i in firstCheck(initial):
        possibleSolutions.append(create(down(i.state), i, i.level + 1))
        possibleSolutions.append(create(up(i.state), i, i.level + 1))
        possibleSolutions.append(create(left(i.state), i, i.level + 1))
        possibleSolutions.append(create(right(i.state), i, i.level + 1))
    for j in possibleSolutions:
        if j.state == None:
            possibleSolutions.remove(j)
    for i in possibleSolutions:
        if i.state == None:
            possibleSolutions.remove(i)
    possibleSolutions.extend(tempSolutions)
    return possibleSolutions


#check for goal, if it matches with goal call it possible parents and add them to list
#call states which is path for goal orderly.
def graph_search(initial_state):
    initial = create(break2Darray(initial_state), initial_parent, initial_level)
    tempSolutionPath = []
    for i in secondCheck(initial):
        if i.state != None:
            if i.state[4] == 0:
                try:
                    tempSolutionPath.append(i.parent.parent)
                except:
                    AttributeError
                try:
                    tempSolutionPath.append(i.parent)
                except:
                    AttributeError
                try:
                    tempSolutionPath.append(i)
                except:
                    AttributeError
                break
    for i in tempSolutionPath:
        if i == None:
            tempSolutionPath.remove(i)

    path = []
    for i in tempSolutionPath:
        if i == None:
            tempSolutionPath.remove(i)
        if i not in path:
            states = []
            states.append([])
            states.append([])
            states.append([])
            states[0].append(i.state[0])
            states[0].append(i.state[1])
            states[0].append(i.state[2])
            states[1].append(i.state[3])
            states[1].append(i.state[4])
            states[1].append(i.state[5])
            states[2].append(i.state[6])
            states[2].append(i.state[7])
            states[2].append(i.state[8])
            path.append(states)
    return path





"""
statesss = [[[0, 4, 5], [6,1,8], [7,3,2]],
            [[5, 0, 4], [6,1,8], [7,3,2]],
            [[5, 4, 0], [6,1,8], [7,3,2]],
            [[5, 4, 6], [0,1,8], [7,3,2]],
            [[5, 4, 1], [6,0,8], [7,3,2]],
            [[5, 4, 8], [6,1,0], [7,3,2]],
            [[5, 4, 7], [6,1,8], [0,3,2]],
            [[5, 4, 3], [6,1,8], [7,0,2]],
            [[5, 4, 2], [6,1,8], [7,3,0]]]

for i in statesss:
    print "*"*50
    print expand(i)
    print "-"*50
    print graph_search(i)
"""











