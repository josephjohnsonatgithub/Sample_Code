# augments the training data according to some criterion
#.   baseline: randomly select
#.   level one: select from most recent of other countries
#.   level two: select according to some network/similarity criterion

import numpy as np
from sklearn.preprocessing import MinMaxScaler


def GetRandomData(agent_X_dict,agent_y_dict,temp_train_index,aug_T,num_feat,y_start_idx):
    agent_list = list(agent_X_dict.keys())
    agent_vec = np.array(agent_list)
    
    aug_X = np.zeros((aug_T,num_feat))
    aug_y = []
    
    for rnd in range(aug_T):
        
        rand_agent_idx = np.random.randint(len(agent_vec))
        temp_agent = agent_vec[rand_agent_idx]
        
        rand_val_idx = np.random.randint(len(temp_train_index))
        temp_y = agent_y_dict[temp_agent][y_start_idx:]
        
        temp_X = agent_X_dict[temp_agent][temp_train_index[rand_val_idx]]
        temp_y = temp_y[temp_train_index[rand_val_idx]]
        
        aug_X[rnd] = temp_X
        aug_y.append(temp_y)
        
    return aug_X, np.array(aug_y)

def GetViolSimData(agent,agent_X_dict,agent_y_dict,temp_train_index,aug_T,num_feat,y_start_idx,viol_dict):
    
    #viol_dict = {agent:vec[:temp_train_index[-1]+1].sum() for agent,vec in flat_d.items()}
    
    agent_list = list(viol_dict.keys())
    viol_vec = np.zeros(len(agent_list))
    agent_vec = np.array(agent_list)
    
    for i,temp_agent in enumerate(agent_list):
        viol_vec[i] = viol_dict[temp_agent]
    
    aug_X = np.zeros((aug_T,num_feat))
    aug_y = []
    
    for rnd in range(aug_T):
        
        sim_agent_idx = agent_list.index(agent)
        while sim_agent_idx == agent_list.index(agent):
            sim_agent_idx = DrawSimilarAgent(viol_dict[agent],viol_vec)
        temp_agent = agent_vec[sim_agent_idx]
        
        rand_val_idx = np.random.randint(len(temp_train_index))
        temp_y = agent_y_dict[temp_agent][y_start_idx:]
        
        temp_X = agent_X_dict[temp_agent][temp_train_index[rand_val_idx]]
        temp_y = temp_y[temp_train_index[rand_val_idx]]
        
        aug_X[rnd] = temp_X
        aug_y.append(temp_y)
        
    return aug_X, np.array(aug_y)

def DrawDifferentAgent(viol_level,viol_vec):
    
    viol_vec = abs(viol_vec - viol_level)
    
    viol_vec = viol_vec.cumsum()
    
    viol_vec  = viol_vec / viol_vec.max()
    
    rand_num = np.random.rand()
    
    return (rand_num < viol_vec).argmax()

def DrawSimilarAgent(viol_level,viol_vec):
    
    viol_vec = abs(viol_vec - viol_level)
    
    viol_vec = abs(viol_vec - viol_vec.max())
    
    viol_vec = viol_vec.cumsum()
    
    viol_vec  = viol_vec / viol_vec.max()
    
    rand_num = np.random.rand()
    
    return (rand_num < viol_vec).argmax()


def GetRegionSimData(agent,
                     agent_X_dict,
                     agent_y_dict,
                     temp_train_index,
                     aug_T,
                     num_feat,
                     y_start_idx,
                     country_region_dict):
    
    member_region = country_region_dict[agent]
    
    # we need this because it specifies only countries being predicted
    region_agents = []
    for country,region in country_region_dict.items():
        if region == member_region and country in agent_X_dict.keys():
            region_agents.append(country)
    
    region_agents.sort()
    
    aug_X = np.zeros((aug_T,num_feat))
    aug_y = []
    
    for rnd in range(aug_T):
        
        rand_agent_idx = np.random.randint(len(region_agents))
        temp_agent = region_agents[rand_agent_idx]
        
        # make this more probable to be recent
        rand_val_idx = GetIdxByProbAscend(temp_train_index)
        
        temp_y = agent_y_dict[temp_agent][y_start_idx:]
        
        #scaler = MinMaxScaler()
        #temp_X = np.array(agent_X_dict[temp_agent])[temp_train_index]
        
        #temp_X = scaler.fit_transform(temp_X)
        
        #temp_X = temp_X[temp_train_index[rand_val_idx]]
        #temp_X = temp_X[rand_val_idx]
        
        temp_X = agent_X_dict[temp_agent][temp_train_index[rand_val_idx]]
        temp_y = temp_y[temp_train_index[rand_val_idx]]
        
        aug_X[rnd] = temp_X
        aug_y.append(temp_y)
        
    return aug_X, np.array(aug_y)

# can mix region with violence or chaos
def GetRegionMixData(agent,agent_X_dict,agent_y_dict,temp_train_index,aug_T,num_feat,y_start_idx,country_region_dict,viol_dict):
    
    country_list = list(viol_dict.keys())
    viol_vec = np.array([viol_dict[country] for country in country_list])
    country_vec = np.array(country_list)
    agent_viol = viol_dict[agent]
    # similar - lowest magnitude will be self
    viol_vec = abs(viol_vec - agent_viol)
    # invert the magnitudes
    #viol_vec = abs(viol_vec - viol_vec.max())
    # scale to 1 being the max
    #print(viol_vec)
    #print(viol_vec.max())
    viol_vec = viol_vec / viol_vec.max()
    
    member_region = country_region_dict[agent]
    region_vec = np.zeros(len(country_list))
    
    # we need this because it specifies only countries being predicted
    for country,region in country_region_dict.items():
        if region == member_region and country in agent_X_dict.keys():
            region_vec[country_list.index(country)] = 1
            
    prob_vec = viol_vec + region_vec
    prob_vec = prob_vec.cumsum()
    prob_vec = prob_vec / prob_vec.max()
    
    aug_X = np.zeros((aug_T,num_feat))
    aug_y = []
    
    for rnd in range(aug_T):
        
        rand_num = np.random.rand()
        temp_agent_idx = (rand_num < prob_vec).argmax()
        temp_agent = country_list[temp_agent_idx]
        #print(country_list[temp_agent_idx])
        
        
        # make this more probable to be recent
        rand_val_idx = np.random.randint(len(temp_train_index)) #GetIdxByProbAscend(temp_train_index)
        #print(rand_val_idx)
        temp_y = agent_y_dict[temp_agent][y_start_idx:]
        #print(agent_y_dict[temp_agent][y_start_idx:])
        temp_X = agent_X_dict[temp_agent][temp_train_index[rand_val_idx]]
        temp_y = temp_y[temp_train_index[rand_val_idx]]
        #print(temp_y)
        aug_X[rnd] = temp_X
        aug_y.append(temp_y)
        
    return aug_X, np.array(aug_y)


def GetIdxByProbAscend(temp_train_index):
    
    arr = np.arange(len(temp_train_index))
    arr = arr.cumsum()
    arr = arr / arr.max()
    rand_num = np.random.rand()
    return (rand_num < arr).argmax()

def GetIdxByProb(arr):
    
    arr = np.arange(len(temp_train_index))
    arr = arr.cumsum()
    arr = arr / arr.max()
    rand_num = np.random.rand()
    return (rand_num < arr).argmax()


def GetNearestNeighbors(agent,
                     agent_X_dict,
                     agent_y_dict,
                     temp_train_index,
                     aug_T,
                     num_feat,
                     y_start_idx,
                     X):
    
    X_vec = np.array(agent_X_dict[agent])[temp_train_index]
    
    ranking = DrawNNSSQ(X_vec[:,:7],X[:,:7],aug_T)
        
    temp_y = agent_y_dict[agent][y_start_idx:]
        
    aug_X = X_vec[ranking]
    aug_y = temp_y[ranking]
        
    return aug_X, aug_y

def DrawRandSimilarVecSSQ(X_vec,X):
    
    ssq_vec = ((X_vec - X)**2).sum(1)
    
    ssq_vec = abs(ssq_vec - ssq_vec.max())
    
    ssq_vec = ssq_vec.cumsum()
    
    ssq_vec  = ssq_vec / ssq_vec.max()
    
    rand_num = np.random.rand()
    
    return (rand_num < ssq_vec).argmax()


def DrawNNSSQ(X_vec,X,T):
    
    ssq_vec = ((X_vec - X)**2).sum(1)
    
    ranking = ssq_vec.argsort()
    
    return ranking[:T]

    

        
        
    
    
    
    
    