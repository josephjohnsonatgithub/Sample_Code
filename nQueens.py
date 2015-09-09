# Eight Queens problem

from copy import deepcopy
from time import sleep
import numpy as np

def criterion(placementList):
	# Receives an eight tuple and decides whether any of the queens can collide
	# By design, there cannot be lateral or horizontal collisions, because we
	# constrain the model not to place queens in the same row or column
	#rowDelta = rowA - rowB
	#colDelat = colA - colB
	# Will be called at the bottom node
	
	for columnA, rowA in enumerate(placementList):
		for columnB, rowB in enumerate(placementList):
			if columnA != columnB:
				if rowA == rowB:
					return False
				rowDelta = abs(rowA - rowB)
				colDelta = abs(columnA - columnB)
				if rowDelta == colDelta:					
					return False
	return True

# the index will have to be the index of the current node
def partialCriterion(placementList, index):
	
	if index == 0:
		return True
	# Very similar to criterion. Checks if solution is correct so far
	for columnA, rowA in enumerate(placementList[:index + 1]):
		for columnB, rowB in enumerate(placementList[:index + 1]):
			if columnA != columnB:
				rowDelta = abs(rowA - rowB)
				colDelta = abs(columnA - columnB)
				if rowDelta == colDelta:					
					return False
	return True
	
# Takes a state class and finds possible immediate successors
def successors(state, index):
	prevIndices = state[:index]
	numSuccessors = len(state) - len(prevIndices)
	successors = []
	
	for i in range(len(state)):
		if i not in prevIndices:
			successors.append(i)
	
	if len(successors) == numSuccessors:
		return successors
	else:
		pass
		
def init_state(n):
	# initial tuple of indices
	l = []
	for i in range(n):
		l.append(0)
	return l
	
def explore(n):
	state = init_state(n)
	return 'number of solutions', intelligentExplore(state, 0)
	
def intelligentExplore(state, index):
	
	successorList = successors(state, index)
	
	num = 0
	
	for idx, succVal in enumerate(successorList):
		currTuple = deepcopy(state)
		currTuple[index] = succVal
			
		if criterion(currTuple):
			num += 1
			print 'solution', currTuple
		elif partialCriterion(currTuple, index):
			num += intelligentExplore(currTuple, index + 1)
		else:
			num += 0
			
	return num
	
print 'Welcome to a state-space approach to solving the n-queens problem'
sleep(1.5)
print 'You will be asked to enter an integer representing the number of queens in your problem.'
sleep(1.5)
nQns = int(raw_input('What will be the size of your board? (recommended 5 < n < 10)\t'))

print explore(nQns)
	
