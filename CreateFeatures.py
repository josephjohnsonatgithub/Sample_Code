import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq, fftn
import math
import os

# for one country, generate S,D,MD,ALL
def GenerateFeaturesForCountry(dict_ndarr_triple_list,
                                                   agent,
                               time_step_end_idx,# need to include up until current time step
                               gamma_dist,
                               log_dist,
                               uniform_dist,w):
    feature_size_list = [temp_ndarr.shape[-1] for (temp_dict,temp_ndarr,data_type) in dict_ndarr_triple_list]
    num_S_features = np.array(feature_size_list).sum()
    S_vec_end_idx_arr = np.array(feature_size_list).cumsum()
    
    trn = 12
    
    # data for fft
    # (time_step_idx-max(time_step_end_idx-w,0):time_step_idx,agent_idx,?)
    time_step_begin_idx = max(time_step_end_idx-w+1,0) # need to increase by 1
    
    # total S features + 3 MD features + total S features (fft transform)
    feature_vec = np.zeros(num_S_features*2+3)
    
    # S
    S_vec_beg_idx = 0
    for i,(temp_agent_idx_dict,temp_embed_ndarr,data_type) in enumerate(dict_ndarr_triple_list):
        feature_vec[S_vec_beg_idx:S_vec_end_idx_arr[i]] = temp_embed_ndarr[time_step_end_idx,temp_agent_idx_dict[agent]]
        S_vec_beg_idx = S_vec_end_idx_arr[i]
    
    if time_step_end_idx < trn:
        print('returning feature_vec early')
        return feature_vec
    
    # MD
    mdarr = np.zeros((time_step_end_idx+1-time_step_begin_idx,num_S_features))
    S_vec_beg_idx = 0
    for i,(temp_agent_idx_dict,temp_embed_ndarr,data_type) in enumerate(dict_ndarr_triple_list):
        mdarr[:,S_vec_beg_idx:S_vec_end_idx_arr[i]] = temp_embed_ndarr[time_step_begin_idx:time_step_end_idx+1,temp_agent_idx_dict[agent],:]
        S_vec_beg_idx = S_vec_end_idx_arr[i]
    feature_vec[num_S_features:num_S_features+3] = MDFFTFit(mdarr,gamma_dist,log_dist,uniform_dist)
    
    # D
    S_vec_idx = 0
    for (temp_agent_idx_dict,temp_embed_ndarr,data_type) in dict_ndarr_triple_list:
        for temp_S_idx in range(temp_embed_ndarr.shape[-1]):
            temp_arr = temp_embed_ndarr[time_step_begin_idx:time_step_end_idx+1,temp_agent_idx_dict[agent],temp_S_idx] # (w,1,1)
            fit = FFTFit(temp_arr,gamma_dist)
            feature_vec[num_S_features+3+S_vec_idx] = fit
            S_vec_idx += 1
    
    return feature_vec

# gets dicts for factions, nodes, graphs
# just parse the structures available
def GetAgentIdxDicts(country_predict_list,country_agent_list):
    # country_agent_list: the order that all country_agents appear in the adjmats and graphs
    # country_faction_set_dict: the order that all country_agents appear in faction data
    
    # factions
    # trivial, but we will do it anyway
    # indexes in faction_emb_ndarr the countries that are predicted on
    faction_country_idx_dict = {agent : i for i,agent in enumerate(country_predict_list)}
    
    # nodes
    # hacky, but it works
    # indexes in node_emb_ndarr the countries that are predicted on
    node_country_idx_dict = {agent : country_agent_list.index(agent) for agent in country_predict_list}
    
    # graphs
    # trivial, but we will do it anyway
    # indexes in graph_emb_ndarr the countries that are predicted on
    graph_country_idx_dict = {agent : 0 for agent in country_predict_list}
    
    return faction_country_idx_dict,node_country_idx_dict,graph_country_idx_dict

# converts the list of factions to a list of the countries that is sorted and used to index all embeddings
def FactionListToCountryList(agent_list):
    country_list = list(set([agent.split('::')[0] for agent in agent_list]))
    country_list.sort()
    return country_list
    

def GetCountryGraphEmbedNDArr(directory_name,dim,agent_idx_dict,country_predict_list):
    data_ndarr = np.zeros(dim)
    
    for temp_dir in os.listdir(directory_name):
        if temp_dir not in country_predict_list:
            continue
        temp_path = directory_name + "/" + temp_dir
        abstract_level_idx = agent_idx_dict[temp_dir]
        temp_df = pd.read_csv(temp_path + "/local_graph.csv",index_col=0,header=0)
        temp_df = temp_df.fillna(0)
        data_ndarr[:,abstract_level_idx,:] = np.array(temp_df)
    
    return data_ndarr

def GetGraphEmbedNDArr(directory_name,
                       dim,
                       graph_idx_dict,
                       graph_node_idx_dict,
                       node_idx_dict,
                       node_graph_dict,
                       predict_list):
    # graph_idx_dict: abstract_level index among all abstract levels
    # node_idx_dict: constituent parts indexes for each abstract level
    # 
    # should be 3-dimensional: (num_time_steps,num_nodes,num_features)
    data_ndarr = np.zeros(dim)
    #print(data_ndarr.shape)
    offset=0
    
    node_graph_dict = {node : {'idx': graph_node_idx_dict[node_graph_dict[node]][node], 
                               'graph': node_graph_dict[node]} 
                       for node in predict_list
                       }
    
    for agent in predict_list:
        graph_dir = node_graph_dict[agent]['graph']
        # index for X array
        node_idx = node_idx_dict[agent]
        graph_path = directory_name + "/" + graph_dir
        
        metric_df = pd.read_csv(graph_path + "/local_graph.csv",index_col=0,header=0)
        metric_df = metric_df.fillna(0)
        metric_arr = np.array(metric_df)
        #print(metric_arr.shape)
        data_ndarr[:,node_idx,:] = metric_arr[:,2].reshape(-1,1) #
    
    return data_ndarr

# 
def GetNodeEmbedNDArr(directory_name,
                      dim,
                      graph_idx_dict,
                      graph_node_idx_dict,
                      node_idx_dict,
                      node_graph_dict,
                      predict_list):
    # graph_idx_dict: abstract_level index among all abstract levels
    # node_idx_dict: constituent parts indexes for each abstract level
    # 
    # should be 3-dimensional: (num_time_steps,num_nodes,num_features)
    data_ndarr = np.zeros(dim)
    offset=0
    
    node_graph_dict = {node : {'idx': graph_node_idx_dict[node_graph_dict[node]][node], 
                               'graph': node_graph_dict[node]} 
                       for node in predict_list
                       }
    
    for agent in predict_list:
        graph_node_idx = node_graph_dict[agent]['idx']
        graph_dir = node_graph_dict[agent]['graph']
        # index for X array
        node_idx = node_idx_dict[agent]
        graph_path = directory_name + "/" + graph_dir
        
        # constituent node feature values
        for i,metric_idx in enumerate([2]): #offset,dim[2]):
            metric_df = pd.read_csv(graph_path + "/" + str(metric_idx) +"_graph.csv",index_col=0,header=0)
            metric_df = metric_df.fillna(0)
            metric_arr = np.array(metric_df)
            data_ndarr[:,node_idx,i] = metric_arr[:,graph_node_idx]
    
    return data_ndarr
    
    
    

    
        
def GenerateFeatures(dict_ndarr_triple_list,w,num_time_steps):
    
    trn=12
    
    # the order that agents appear
    agent_list = list(dict_ndarr_triple_list[0][0].keys())
    
    agent_X_dict = {agent:[] for agent in agent_list}
    
    np.set_printoptions(precision=3, linewidth=200, suppress=True)
    
    gamma_dist,log_dist,uniform_dist = GenerateDistributions()
    
    # go right through rows, no need to worry about column
    feature_arr_list = []
    country_data_list = []
    
    for time_step in range(num_time_steps):
        if time_step < trn:
            continue
        for agent in agent_list:
            t_w = w#int(w*chaos_dict[agent])
            #print(agent,t_w)
            t_feature_vec = GenerateFeaturesForCountry(dict_ndarr_triple_list,
                                                           agent,
                                                           time_step,
                                                           gamma_dist,
                                                           log_dist,
                                                           uniform_dist,t_w)
            agent_X_dict[agent].append(t_feature_vec)
    
    return agent_X_dict


def GetEmbArrFromCsv(data_dir,dim,node_emb_idx_dict,data_slice):
 
    data_ndarr = np.zeros(dim)
    
    for doc in os.listdir(data_dir):
        
        temp_data = doc.split(".")
        if 'ipynb_checkpoints' in temp_data:
            continue
        year = int(temp_data[0])
        month = int(temp_data[1])
        temp_df = pd.read_csv(data_dir + "/" + doc,index_col=0,header=0)
        temp_df = temp_df.fillna(0)
        #data_ndarr[node_emb_idx_dict[year][month],:] = np.array(temp_df[data_slice]) / (abs(np.array(temp_df[data_slice])).max(0)+.000000001)
        data_ndarr[node_emb_idx_dict[year][month],:] = temp_df[data_slice]
    
    return data_ndarr
        
def GetYearMonthIdxDict(year_range):
    d = {}
    
    idx = 0
    for year in year_range:
        d.setdefault(year,{})
        for month in range(1,13):
            d[year][month] = idx
            idx += 1
    return d
            
            
def GenerateDistributions():
    
    # gamma dist
    gammaData = np.random.gamma(2,2,size=1000000)
    gammaDist = np.histogram(gammaData,bins=128)
    gammaDistNorm = gammaDist[0]/gammaDist[0].sum() #.cumsum()/1000

    # log normal
    logNormData = np.random.lognormal(0,.8,size=1000000)
    logNormDist = np.histogram(logNormData,bins=128)
    logNormDistNorm = logNormDist[0]/logNormDist[0].sum() #.cumsum()/1000
    
    # uniform dist
    uniformDistNorm = np.ones(128) * (1/128)
    
    return gammaDistNorm,logNormDistNorm,uniformDistNorm

def ApplyWelchWindow(arr):
    
    narr = np.zeros(len(arr))
    
    for n,tok in enumerate(arr):
        narr[n] = Welch(n,arr)*tok
        
    return narr
 
def Welch(n,arr):
    
    Wn = 1 - ((n-.5*(len(arr)-1))/(.5*(len(arr)+1)))**2
    
    return Wn
            
# takes a window of data to apply to fft
def FFTFit(arr,gammaDistNorm):
    
    arr = ApplyWelchWindow(arr)
    
    array_len = len(arr)
    fitted_coeff = np.polyfit(np.arange(len(arr))+1,arr,1)
    trend = np.linspace(1,array_len,array_len)*fitted_coeff[0] + fitted_coeff[1]
    arr_no_trend = arr - trend
  
    ## fft ##
    arr_fft = Trans(arr_no_trend)
    lV = np.array(arr_fft[:128])
    lV = lV / (lV.sum() + .00000000000000000001)

    ## fit ##
    gammaDiff = lV - gammaDistNorm
    gammaSS = math.sqrt(np.multiply(gammaDiff,gammaDiff).sum()/128)
    
    return gammaSS
    
def MDFFTFit(ndarr,gammaDistNorm,logNormDistNorm,uniformDistNorm):
    ndarr_welch = np.zeros((ndarr.shape[0],ndarr.shape[1]))
    for col in range(ndarr.shape[1]):
        ndarr_welch[:,col] = ApplyWelchWindow(ndarr[:,col])
    
    xfL,yfL = TransND(ndarr_welch)
    lV = np.nan_to_num(np.array(yfL[:128,0]),0)
    lV = lV / lV.sum()
    gammaDiff = lV - gammaDistNorm
    logNormDiff = lV - logNormDistNorm
    uniformDiff = lV - uniformDistNorm
    gammaSS = math.sqrt(np.multiply(gammaDiff,gammaDiff).sum()/128)
    logNormSS = math.sqrt(np.multiply(logNormDiff,logNormDiff).sum()/128)
    uniformSS = math.sqrt(np.multiply(uniformDiff,uniformDiff).sum()/128)
    return np.array([gammaSS,uniformSS,logNormSS])

def Trans(vec):
    yf = fft(vec,n=256)
    
    return np.abs(yf).tolist()

def TransND(vec):
    
    yf = fftn(vec,s=(256,256))
    xf = fftfreq(256,d=1)
    
    return xf,np.abs(yf)
