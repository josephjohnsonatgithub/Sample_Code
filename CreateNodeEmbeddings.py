# converts the action data to a graph

import pandas as pd
import numpy as np
import datetime
import codecs
from copy import deepcopy
import importlib
import math
from collections import defaultdict
import networkx as nx
import time
import os

from sklearn.preprocessing import MinMaxScaler

import FairGood as fgc
import CreateGraphs as cg
import PropagationII as prop2

importlib.reload(cg)
importlib.reload(prop2)


    
def ReverseDict(d,country_predict_list):
    rev_d = {}
    for key,val in d.items():
        if val in country_predict_list:
            rev_d.setdefault(val,set())
            rev_d[val].add(key.split('::')[1])
    return rev_d

def FactListToPredFactSetDict(faction_agent_list,country_predict_list):
    country_fact_set_dict = {country:set() for country in country_predict_list}
    
    for faction in faction_agent_list:
        [country,agent] = faction.split("::")
        if country in country_predict_list:
            country_fact_set_dict[country].add(agent)
            
    return country_fact_set_dict
    


# for now, will only return values for active nodes    
def GenerateEmbeddings(adjmat_graph_dict,
                       agent_idx_dict,
                       abstract_level_name,
                       country_predict_list,
                       nmc_dict,
                      num_time_steps,
                      year_dict):
    abstract_level_agent_list = list(agent_idx_dict.keys())
    abstract_level_agent_list.sort()
    meta_idx_dict = {abs_agent:idx for idx,abs_agent in enumerate(abstract_level_agent_list)}
    prop_dict = {}
    
    for abstract_level,temp_adjmat in adjmat_graph_dict.items():
        print(abstract_level)
        # every abstract level has a graph and nodes array
        temp_agent_idx_dict = agent_idx_dict[abstract_level]
        temp_idx_agent_dict = {idx:agent for agent,idx in temp_agent_idx_dict.items()}
        agent_list = []
        for i in range(len(temp_idx_agent_dict.keys())):
            agent_list.append(temp_idx_agent_dict[i])
        graph_emb_arr = np.zeros((num_time_steps,13))
        node_emb_arr = np.zeros((num_time_steps,6,len(agent_list)))
                    
        #prop_dict.setdefault(abstract_level,prop2.PropagationII(temp_agent_list))
        #moral_graph(G)[source]
        #rich_club_coefficient(G, normalized=True, Q=100, seed=None)
        for time_step in range(num_time_steps): 
            #print(time_step)
            
            curr_mat = temp_adjmat[max(0,time_step-12):time_step+1,:,:].sum(0)
            
            if abs(curr_mat).sum()==0:
                graph_emb_arr[time_step,:]=0
                #print("no actions")
            else:
                curr_mat = curr_mat / abs(curr_mat).max()

                G = GetGraphFromAdjMat(curr_mat,temp_agent_idx_dict,nmc_dict[year_dict[time_step]]) # needs to be the previous year
                curr_mat,G,temp_agent_list,temp_agent_idx_list = TrimStructures(G,curr_mat,agent_list)
                #print('after trim',curr_mat.shape)

                wght_d = dict(G.nodes(data="weight"))
                label_d = dict(G.nodes(data="label"))

                # both active and non-active agents
                n_wgt = AssignGraphValues(wght_d,G,temp_agent_list)
                n_wgt = n_wgt / abs(n_wgt).max()
                
                
                
                f,g = fgc.compute_fairness_goodness(G)

                # graph level
                f_arr = AssignGraphValues(f,G,temp_agent_list) # ndarray(num_agents,1)
                g_arr = AssignGraphValues(g,G,temp_agent_list) # ndarray(num_agents,1)
                # node level
                AssignNodeValues(f_arr,temp_agent_idx_list,node_emb_arr[time_step,0,:])
                AssignNodeValues(g_arr,temp_agent_idx_list,node_emb_arr[time_step,1,:])
                #print(f_arr.shape,g_arr.shape)
                
                
                exnet_arr = ExNetNodeVals(curr_mat,temp_agent_list)
                
                for col in range(exnet_arr.shape[1]):
                    AssignNodeValues(exnet_arr[:,col].transpose()
                                     ,temp_agent_idx_list,
                                     node_emb_arr[time_step,col+2,:])
                
                
                # graph level
                f_corr = NodeWgtCorr(f_arr,n_wgt)
                g_corr = NodeWgtCorr(g_arr,n_wgt)
                #fg_corr = NodeWgtCorr(fg_arr,n_wgt)
                
                
                struct_bal = StructBal(curr_mat)
                
                assort_list = Assort(curr_mat)

                exnet_corr_list = ExNetGraphVals(exnet_arr,n_wgt) # four features

                graph_emb_arr[time_step,:] = np.array([g_corr,f_corr,struct_bal] 
                                                      + assort_list 
                                                      + exnet_corr_list)
                
                graph_directory = abstract_level_name + "GraphData/" + abstract_level

                if not os.path.exists(graph_directory):
                    os.makedirs(graph_directory)

                pd.DataFrame(graph_emb_arr).to_csv(graph_directory + "/local_graph.csv")
                
                node_directory = abstract_level_name + "NodeData/" + abstract_level

                if not os.path.exists(node_directory):
                    os.makedirs(node_directory)

                
                for metric_idx in range(6):
                    pd.DataFrame(node_emb_arr[:,metric_idx,:]).to_csv(node_directory + "/" + str(metric_idx) + "_graph.csv")
                    
                
def AssignNodeValues(arr,agent_idx_list,node_emb_arr):
    
    # arr: [value for idx=0 in agent_idx_list, val for idx=1 in agent_idx_list,...]
    
    for i,val in enumerate(agent_idx_list):
        node_emb_arr[val] = arr[i]
    


def AssignGraphValues(value_dict,G,agent_list):
    # |agent_list| >= |value_dict|
    
    agent_value_arr = np.zeros(len(agent_list))
    
    label_d = dict(G.nodes(data="label"))
    
    label_list = [val for key,val in label_d.items()]
    value_arr = np.array([value_dict[key] for key,val in label_d.items()])
    
    for idx,label in enumerate(label_list):
        agent_value_arr[agent_list.index(label)] = value_arr[idx]
        
    return agent_value_arr
        
    
# gets list as well as trimmed matrix and trimmed graph
def TrimStructures(G,mat,agent_list):

    trim_mat,trim_agent_list,trim_agent_idx_list = TrimAdjMat(mat,agent_list)
    
    trim_G, trim_label_list  = TrimGraph(G)
    
    #print('assert lists', trim_agent_list == trim_label_list)
    
    return trim_mat,trim_G,trim_agent_list,trim_agent_idx_list


def TrimAdjMat(mat,agent_list):
      
    agent_arr = np.array(agent_list)
    agent_idx_arr = np.arange(len(agent_list))
    
    inactive_row = (abs(mat).sum(1) == 0)
    inactive_col = (abs(mat).sum(0) == 0)
    delete_row = [i for i,x in enumerate(inactive_row) if x == True]
    delete_col = [i for i,x in enumerate(inactive_col) if x == True]
    
    delete_idx_list = list(set(delete_row).intersection(set(delete_col)))
    
    trim_mat = np.delete(mat,delete_idx_list,0)
    trim_mat = np.delete(trim_mat,delete_idx_list,1)
    
    trim_agent_arr = np.delete(agent_arr,delete_idx_list)
    trim_agent_idx_arr = np.delete(agent_idx_arr,delete_idx_list)
    
    return trim_mat, list(trim_agent_arr), list(trim_agent_idx_arr)


def TrimGraph(G):
    # https://newbedev.com/python-networkx-remove-nodes-and-edges-with-some-condition#:~:text=python%20networkx%20remove%20nodes%20and%20edges%20with%20some,compactly%20create%20a%20list%20of%20nodes%20to%20delete.
    
    remove = [node for node,degree in dict(G.degree()).items() if degree == 0]
    
    trim_G = deepcopy(G)
    trim_G.remove_nodes_from(remove)
    
    trim_label_d = dict(trim_G.nodes(data="label"))
    trim_label_list = [val for key,val in trim_label_d.items()]
    
    return trim_G,trim_label_list

def NodeWgtCorr(node_feat_arr,node_wgt_arr):
    feat_corr = np.corrcoef(node_feat_arr,node_wgt_arr)[0,1]
    
    if not np.isfinite(feat_corr):
        feat_corr = 0.0
    
    return feat_corr

def StructBal(mat):
    
    den = 0
    num = 0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[0])[i+1:]:
            for k in range(mat.shape[0])[j+1:]:
                tVal = mat[i,j] * mat[j,k] * mat[k,i]
                num += tVal #max(tVal,0)
                den += abs(tVal)
                tVal = mat[i,k] * mat[k,j] * mat[j,i]
                num += tVal #max(tVal,0)
                den += abs(tVal)
    return num/(den + .0000000000001)

# gets the correlation scores according to node scores
def ExNetGraphVals(node_embed_mat,node_wgt_arr):
    l = []
    for exnet_type in range(node_embed_mat.shape[1]):
        t_node_feat_arr = node_embed_mat[:,exnet_type]
        l.append(NodeWgtCorr(t_node_feat_arr,node_wgt_arr))
                 
    return l
        
def ExNetNodeVals(mat,agent_list):  
    
    # fine to rescale
    mat = mat + mat.transpose()
    mat = mat / abs(mat).max()
    
    # get the largest positive eigenvalue
    mat = np.nan_to_num(mat, nan=0, posinf=1, neginf=-1)
    evm = np.abs(np.linalg.eig(mat)[0].max())
    
    ones_mat = np.ones((mat.shape[0],1))
    alpha = 1
    t_mat = np.zeros((mat.shape[0],1)) # (active agents,1)
    node_embed_mat = np.zeros((len(agent_list),4)) # (total agents,4)
    
    for i,beta in enumerate([-1,1,.5,-.5]): # bargaining too close
        #print(i,beta) #-.9,-.8,-.7,-.6,-.5,-.4,-.3,-.2,-.1,0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1]:
        if beta < (1/np.abs(evm)): 
            try:
                t_mat = alpha * np.linalg.inv(np.eye(mat.shape[0]) - (beta * mat)) @ mat @ ones_mat
                t_mat = np.asarray(t_mat)
            except:
                t_mat = (np.zeros((mat.shape[0],1)))
        else:
            try:
                t_mat = - alpha * np.linalg.inv(np.eye(mat.shape[0]) - (beta * mat)) @ mat @ ones_mat
                t_mat = np.asarray(t_mat)
            except:
                t_mat = np.zeros((mat.shape[0],1))
        t_mat = np.nan_to_num(t_mat, nan=0, posinf=1, neginf=-1)
        #scaler = MinMaxScaler(feature_range=(-1,1))
        #t_mat = scaler.fit_transform(t_mat.reshape(-1,1))
        t_mat = t_mat.flatten()
        t_mat = t_mat / abs(t_mat).max()
        node_embed_mat[:,i] = t_mat
         
    return node_embed_mat    

# calculates cluster coefficient for one node on a signed graph with weighted edges
def ClustNodeCoeff(graph):
    graph = graph + graph.transpose()

    graph = graph/abs(graph).max()
    l = []

    for n_i in range(graph.shape[0]):
        num = 0.0
        den = 0.0
        # node to calculate coeff for
        for n_j in range(graph.shape[0]):
            if (n_j == n_i):
                continue
            for n_k in range(graph.shape[0]):
                if (n_k == n_i) or (n_k == n_j):
                    continue
                num += graph[n_i,n_j] * graph[n_j,n_k] * graph[n_k,n_i]
                den += abs(graph[n_i,n_j] * graph[n_k,n_i])
        try:
            l.append(np.nan_to_num(num/den,nan=1))
        except:
            l.append(1)
            
    scVec = np.array(l)
    return scVec #np.corrcoef(scVec,np.array(pV.transpose()))[0,1] #, np.array(scVec.tolist())

def FielderEigVal(A):
    
    A = A + A.transpose()
    A = A/abs(A).max()
    
    # in-degree for graph Laplacian
    D = np.multiply(np.eye(A.shape[0]),A.sum(1))
    
    L = D - A
    
    eigVals = np.array([])
    eigVecs = np.array([])
    try:
        eigVals,eigVecs = np.linalg.eig(L)
    except:
        return 0
    
    # largest positive eigenvalue per (paper)
    # try abs or real
    for idx in eigVals.argsort():
        num = np.abs(eigVals[idx])
        # vector useful for clustering
        #vecs = np.abs(eigVecs[:,idx])
        if abs(num) < 0.00001:
            continue
        else:
            return num

    return 0

def Assort(graph):
    
    graph = graph + graph.transpose()
    
    posMat = (graph>0)
    negMat = (graph<0)
    l = []
    
    N = graph.shape[0]

    M = -1
    jkMult = 0
    jkSum = 0
    jkSumSq = 0

    

    #print("r^+(+,+)")
    M = posMat.sum()/2
    for j in range(N-1):
        for k in range(j+1,N):
            if (posMat[j,k] == 0):
                continue
            kDeg = posMat[k,:].sum()
            jDeg = posMat[j,:].sum()
            jkMult += kDeg * jDeg
            jkSum += kDeg + jDeg
            jkSumSq += (kDeg ** 2) + (jDeg ** 2)
    arg1Num = 1/M * jkMult
    arg2Num = (1/M * 1/2 * jkSum)**2
    arg1Den = 1/M * 1/2 * jkSumSq
    arg2Den = (1/M * 1/2 * jkSum)**2
    l.append(np.nan_to_num((arg1Num-arg2Num)/(arg1Den-arg2Den),nan=0))
    
    #print("\n^-(+,+)")
    M = negMat.sum()/2
    jkMult = 0
    jkSum = 0
    jkSumSq = 0

    for j in range(N-1):
        for k in range(j+1,N):
            if (negMat[j,k] == 0):
                continue
            kDeg = posMat[k,:].sum()
            jDeg = posMat[j,:].sum()
            jkMult += kDeg * jDeg
            jkSum += kDeg + jDeg
            jkSumSq += (kDeg ** 2) + (jDeg ** 2)
    arg1Num = 1/M * jkMult
    arg2Num = (1/M * 1/2 * jkSum)**2
    arg1Den = 1/M * 1/2 * jkSumSq
    arg2Den = (1/M * 1/2 * jkSum)**2
    l.append(np.nan_to_num((arg1Num-arg2Num)/(arg1Den-arg2Den),nan=0))

    #print("\n^+(-,-)")
    M = posMat.sum()/2
    
    jkMult = 0
    jkSum = 0
    jkSumSq = 0

    for j in range(N-1):
        for k in range(j+1,N):
            if (posMat[j,k] == 0):
                continue
            jDeg = negMat[j,:].sum()
            kDeg = negMat[k,:].sum()
            jkMult += kDeg * jDeg
            jkSum += kDeg + jDeg
            jkSumSq += (kDeg ** 2) + (jDeg ** 2)
    arg1Num = 1/M * jkMult
    arg2Num = (1/M * 1/2 * jkSum)**2
    arg1Den = 1/M * 1/2 * jkSumSq
    arg2Den = (1/M * 1/2 * jkSum)**2
    l.append(np.nan_to_num((arg1Num-arg2Num)/(arg1Den-arg2Den),nan=0))

    #print("\n^-(-,-)")
    M = negMat.sum()/2
    jkMult = 0
    jkSum = 0
    jkSumSq = 0

    for j in range(N-1):
        for k in range(j+1,N):
            if (negMat[j,k] == 0):
                continue
            jDeg = negMat[j,:].sum()
            kDeg = negMat[k,:].sum()
            jkMult += kDeg * jDeg
            jkSum += kDeg + jDeg
            jkSumSq += (kDeg ** 2) + (jDeg ** 2)
    arg1Num = 1/M * jkMult
    arg2Num = (1/M * 1/2 * jkSum)**2
    arg1Den = 1/M * 1/2 * jkSumSq
    arg2Den = (1/M * 1/2 * jkSum)**2
    l.append(np.nan_to_num((arg1Num-arg2Num)/(arg1Den-arg2Den),nan=0))

    #print("\n^+(+,-)")
    M = posMat.sum()/2
    jkMult = 0
    jkSum = 0
    jkSumSq = 0

    for j in range(N-1):
        for k in range(j+1,N):
            if (posMat[j,k] == 0):
                continue
            jDeg = posMat[j,:].sum()
            kDeg = negMat[k,:].sum()
            jkMult += kDeg * jDeg
            jkSum += kDeg + jDeg
            jkSumSq += (kDeg ** 2) + (jDeg ** 2)
    arg1Num = 1/M * jkMult
    arg2Num = (1/M * 1/2 * jkSum)**2
    arg1Den = 1/M * 1/2 * jkSumSq
    arg2Den = (1/M * 1/2 * jkSum)**2
    l.append(np.nan_to_num((arg1Num-arg2Num)/(arg1Den-arg2Den),nan=0))

    #print("\n^-(+,-)")
    M = negMat.sum()/2
    jkMult = 0
    jkSum = 0
    jkSumSq = 0

    for j in range(N-1):
        for k in range(j+1,N):
            if (negMat[j,k] == 0):
                continue
            jDeg = posMat[j,:].sum()
            kDeg = negMat[k,:].sum()
            jkMult += kDeg * jDeg
            jkSum += kDeg + jDeg
            jkSumSq += (kDeg ** 2) + (jDeg ** 2)
    arg1Num = 1/M * jkMult
    arg2Num = (1/M * 1/2 * jkSum)**2
    arg1Den = 1/M * 1/2 * jkSumSq
    arg2Den = (1/M * 1/2 * jkSum)**2
    l.append(np.nan_to_num((arg1Num-arg2Num)/(arg1Den-arg2Den),nan=0))

    return l


def GetGraphFromAdjMat(mat,agent_idx_dict,nmc_dict):
    # reverse the keys,val in the idx_dict
    idx_agent_dict = {val:key for key,val in agent_idx_dict.items()}
    
    G = nx.DiGraph()
    
    pop = mat.sum(0) #GetNMCPopArr(idx_agent_dict,nmc_dict)

    for i in range(mat.shape[0]):
        G.add_node(i,weight = pop[i],label=idx_agent_dict[i])
       
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            wght = mat[i,j]
            # do not include self-reference
            if wght != 0 and i != j:
                if wght < 0:
                    G.add_edge(i,j,color='r',weight=wght) 
                else:
                    G.add_edge(i,j,color='g',weight=wght)
    return G   

def GetNMCPopArr(idx_agent_dict,nmc_dict):
    
    pop_arr = np.zeros(len(idx_agent_dict))
    
    for idx,agent in idx_agent_dict.items():
        agent_country=agent.split('::')[0]
        if agent_country in nmc_dict.keys():
            pop_arr[idx] = nmc_dict[agent_country]
    return pop_arr