# takes the adj matrix B at each round t and learns which parameters worked best for round t+1
# predict actions
# estimates the propagation for all agents

import numpy as np
import pandas as pd
from copy import deepcopy
from scipy import optimize

class PropagationII:
    
    def __init__(self, agents):
        # currMat is a pandas df; (helps to get the labels)
        self.B             = None
        self.agents     = agents
        self.numAgents    = len(agents)
        # direct propagation:     M = B
        self.alpha_dp     = 0.0
        self.dpMat        = None
        # co-citation:             M = B^T B
        self.alpha_cc     = 0.0
        self.ccMat        = None
        # transpose trust:         M = B^T
        self.alpha_tt     = 0.0
        self.ttMat        = None
        # trust coupling:        M = B B^T
        self.alpha_tc     = 0.0
        self.tcMat        = None
        # tit-for-tat
        self.alpha_tft    = 0.0
        self.tftMat        = None
        self.pMat        = None
        self.masterXL    = []
        self.masteryL    = []
        
    # will be a numpy array
    def ResetData(self, adjMat):
        # the action matrix that will be propagated
        self.B              = deepcopy(adjMat)
        #print('B',self.B.shape,'B_act',self.B_act.shape)
        #for i in range(len(self.B)):
        #    self.B[i,i] = 0
        B                   = deepcopy(self.B)
        # set M for each operator matrix
        self.a_sum          = 0.0
        self.dp_sum         = 0.0
        self.dpMat          = B
        self.cc_sum         = 0.0
        self.ccMat          = B.transpose() @ B
        self.tt_sum         = 0.0
        self.ttMat          = B.transpose()
        self.tc_sum         = 0.0
        self.tcMat          = B @ B.transpose()
        self.tft_sum        = 0.0
        self.p_sum          = 0.0
        self.mse            = 0.0
        
        self.FinalizeMatrices()
        
    def AppendXData(self,adjMat):
        #print('TRAIN DATA')
        XL,yL = self.AdjMatToPredMat(adjMat)
        self.masterXL.extend(XL)
        
    def AppendYData(self,adjMat):
        XL,yL = self.AdjMatToPredMat(adjMat)
        self.masteryL.extend(yL)
    
    # takes in matrix
    def LearnParams(self): #, row):
        trXMat = self.masterXL[-4*(self.numAgents-1)*(self.numAgents-1):]
        tryMat = self.masteryL[-4*(self.numAgents-1)*(self.numAgents-1):]
        
        # data up to latest round
        trainXMat = np.array(trXMat)
        trainyMat = np.array(tryMat)
        
        return self.CalcParamsMSE(trainXMat,trainyMat)

        
    def CalcParamsMSE(self,trXMat,tryMat):#,teXMat,teyMat):
        vals,te = optimize.nnls(trXMat,tryMat)
        
        return vals/vals.sum()
        
        
    def AdjMatToPredMat(self,adjMat):
        #print('adjMat\n',adjMat,'\n')
        XL = []; yL = []
        self.a_sum     = abs(adjMat).sum(1)
        yVec = -1
        xVec = -1
        
        for idx in range(self.numAgents):
            for row in range(self.numAgents):
                if row == idx:
                    continue
                indices = (row,idx)
                XVec = [1,self.dpMat[indices],self.ccMat[indices],self.tftMat[indices]]
  #[1,self.dpMat[indices],self.ccMat[indices],self.ttMat[indices],self.tcMat[indices],self.tftMat[indices],self.pMat[indices]]
                #print(adjMat[indices])
                yVec = np.nan_to_num(adjMat[indices] / self.a_sum[row])
                XL.append(XVec)
                yL.append(yVec)
        return XL,yL
        
        
    def FinalizeMatrices(self):
        # finalize the matrices: B M
        self.dpMat          =   (self.B @ self.dpMat) 
        self.dp_sum         =   abs(self.dpMat).sum(1).reshape(-1,1)
        self.dpMat          =   np.nan_to_num(self.dpMat / self.dp_sum)
        # always defect
        self.ccMat         = (self.B @ self.ccMat)  
        self.cc_sum       = abs(self.ccMat).sum(1).reshape(-1,1)
        self.ccMat        = np.nan_to_num(self.ccMat / self.cc_sum)
        #print('co-citation\n', self.ccMat)
        self.ttMat        = (self.B @ self.ttMat)
        self.tt_sum       = abs(self.ttMat).sum(1).reshape(-1,1)
        self.ttMat        = np.nan_to_num(self.ttMat / self.tt_sum)
        #print('transpose trust\n', self.ttMat)
        self.tcMat         = (self.B @ self.tcMat)
        self.tc_sum        = abs(self.tcMat).sum(1).reshape(-1,1)
        self.tcMat        = np.nan_to_num(self.tcMat / self.tc_sum)
        #print('trust coupling\n', self.tcMat)
        self.tftMat     = (self.B.transpose())
        self.tft_sum    = abs(self.tftMat).sum(1).reshape(-1,1) # + (np.random.rand() * .01)
        self.tftMat        = np.nan_to_num(self.tftMat / self.tft_sum)
        # is now all_defect
        self.pMat         =  deepcopy(self.B)
        #print('B\n',self.B.round(2))
        self.p_sum        = abs(self.pMat).sum(1).reshape(-1,1)
        self.pMat        = np.nan_to_num(self.pMat / self.p_sum)
        self.b_sum         = abs(self.B).sum(1).reshape(-1,1)
    
