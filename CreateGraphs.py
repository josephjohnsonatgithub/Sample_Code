import pandas as pd
import numpy as np
import datetime
import os
import networkx as nx

# generates the graphs in sequential order for ICEWS
# gets event dicts at faction, country, and region levels
def GetAllEventDictFromDf(year_df_dict,country_region_dict,country_predict_list):
    # {year,{month,{global,{src_country,{tgt_country:val,...}}}}} 5 levels, but one trivial ('global')
    global_event_dict = {year:{month:{} for month in range(1,13)} for year in year_df_dict.keys()}
    # {year,{month,{region,{src_country,{tgt_country:val,...}}}}} 5 levels like CEDS::Cuba
    region_event_dict = {year:{month:{} for month in range(1,13)} for year in year_df_dict.keys()}
    # {year,{month,{country,{src_faction,{tgt_faction:val,...}}}}} 5 levels
    country_event_dict = {year:{month:{} for month in range(1,13)} for year in year_df_dict.keys()}
    
    for year,temp_df in year_df_dict.items():
        for month in range(1,13):
            print('year:',year,'month',month)
            temp_global_d,temp_region_d,temp_country_d = GetTempEventDictFromDf(temp_df,year,month,
                                                                                country_region_dict,
                                                                               country_predict_list)
            global_event_dict[year][month] = temp_global_d
            region_event_dict[year][month] = temp_region_d
            country_event_dict[year][month] = temp_country_d
    
    return global_event_dict,region_event_dict,country_event_dict
            
        
def GetTempEventDictFromDf(df,year,month,country_region_dict,country_predict_list):
    temp_df = df[(df['Event Date'].dt.year == year) & (df['Event Date'].dt.month == month)]
    temp_df.reset_index(drop=True,inplace=True)
    temp_global_d,temp_region_d,temp_country_d = GetDictByLevel(temp_df,
                                                                country_region_dict,
                                                                country_predict_list)
    
    return temp_global_d,temp_region_d,temp_country_d
        
# will be non-truncated
def GetDictByLevel(df,country_region_dict,country_predict_list):
    temp_global_event_dict = {}
    temp_region_event_dict = {}
    temp_country_event_dict = {}
    
    for row in range(df.shape[0]):
        
        source_name = df['Source Name'][row]
        source_sectors = df['Source Sectors'][row]
        source_country = df['Source Country'][row]
        source_agent_sectors = [source_country+"::"+source_agent for source_agent in source_sectors.split(',')]
        
        target_name = df['Target Name'][row]
        target_sectors = df['Target Sectors'][row]
        target_country = df['Target Country'][row]
        target_agent_sectors = [target_country+"::"+target_agent for target_agent in target_sectors.split(',')]
        
        val = df['Intensity'][row]
        
        UpdateGlobalDict(temp_global_event_dict,
                         source_name,source_agent_sectors,source_country,
                         target_name,target_agent_sectors,target_country,
                         val)
        UpdateRegionDict(temp_region_event_dict,
                         source_name,source_agent_sectors,source_country,
                         target_name,target_agent_sectors,target_country,
                         val,
                        country_region_dict)
        UpdateCountryDictLocal(temp_country_event_dict,
                          source_name,source_agent_sectors,source_country,
                         target_name,target_agent_sectors,target_country,
                         val,
                         country_predict_list)
        
    return temp_global_event_dict,temp_region_event_dict,temp_country_event_dict
        
def UpdateGlobalDict(d,source_name,source_agent_sectors,source_country,
                     target_name,target_agent_sectors,target_country,
                     val):
    d.setdefault('global',{})
    d['global'].setdefault(source_country,{})
    d['global'][source_country].setdefault(target_country,0.0)
    d['global'][source_country][target_country] += val
      
         
    # alternate approach
    #for source_agent in source_agent_sectors:
    #    for target_agent in target_agent_sectors:
    #        d['global'].setdefault(source_agent,{})
    #        d['global'][source_agent].setdefault(target_agent,0.0)
    #        d['global'][source_agent][target_agent] += val

# only operates at country level
def UpdateRegionDict(d,source_name,source_agent_sectors,source_country,
                     target_name,target_agent_sectors,target_country,
                     val,country_region_dict):
    # can be consolidated, but I don't care right now
    if (source_country in country_region_dict.keys()) and (target_country in country_region_dict.keys()):
        source_region = country_region_dict[source_country]
        target_region = country_region_dict[target_country]
        
        # source region will always get a transaction in this case
        d.setdefault(source_region,{})
        d[source_region].setdefault(source_country,{})
        d[source_region][source_country].setdefault(target_country,0.0)
        d[source_region][source_country][target_country] += val
        
        # different regions, count the transaction in both regions
        if source_region != target_region:
            d.setdefault(target_region,{})
            d[target_region].setdefault(source_country,{})
            d[target_region][source_country].setdefault(target_country,0.0)
            d[target_region][source_country][target_country] += val
            
        
    elif (source_country in country_region_dict.keys()) and (target_country not in country_region_dict.keys()):
        source_region = country_region_dict[source_country]
        
        # only source region gets the transaction
        d.setdefault(source_region,{})
        d[source_region].setdefault(source_country,{})
        d[source_region][source_country].setdefault(target_country,0.0)
        d[source_region][source_country][target_country] += val 
    
    elif (source_country not in country_region_dict.keys()) and (target_country in country_region_dict.keys()):
        
        target_region = country_region_dict[target_country]
        
        # only target region gets the transaction
        d.setdefault(target_region,{})
        d[target_region].setdefault(source_country,{})
        d[target_region][source_country].setdefault(target_country,0.0)
        d[target_region][source_country][target_country] += val 
        

# only considers local interactions
def UpdateCountryDictLocal(d,
                      source_name,source_agent_sectors,source_country,
                      target_name,target_agent_sectors,target_country,
                      val,country_predict_list):
    
    for source_agent in source_agent_sectors:
        for target_agent in target_agent_sectors:
            if source_country in country_predict_list and (source_country == target_country):
                d.setdefault(source_country,{})
                d[source_country].setdefault(source_agent,{})
                d[source_country][source_agent].setdefault(target_agent,0.0)
                d[source_country][source_agent][target_agent] += val
        
# considers all agents a country comes in contact with       
def UpdateCountryDict(d,
                      source_name,source_agent_sectors,source_country,
                      target_name,target_agent_sectors,target_country,
                      val,country_predict_list):
    
    for source_agent in source_agent_sectors:
        for target_agent in target_agent_sectors:
            if source_country in country_predict_list:
                d.setdefault(source_country,{})
                d[source_country].setdefault(source_agent,{})
                d[source_country][source_agent].setdefault(target_agent,0.0)
                d[source_country][source_agent][target_agent] += val
            
            if (source_country != target_country) and (target_country in country_predict_list):
                d.setdefault(target_country,{})
                d[target_country].setdefault(source_agent,{})
                d[target_country][source_agent].setdefault(target_agent,0.0)
                d[target_country][source_agent][target_agent] += val
    
def GetAllIdxDict(global_event_dict,region_event_dict,country_event_dict):
    
    global_idx_dict = GetIdxDict(global_event_dict)
    region_idx_dict = GetIdxDict(region_event_dict)
    country_idx_dict = GetIdxDict(country_event_dict)
    
    return global_idx_dict,region_idx_dict,country_idx_dict

def GetIdxDict(d):
    abstract_level_agent_set_dict = {}
    
    for year,year_dict in d.items():
        for month,month_dict in year_dict.items():
            # this is the level of {global,region,country}
            for abstract_level,abstract_level_dict in month_dict.items():
                abstract_level_agent_set_dict.setdefault(abstract_level,set())
                for source,source_dict in abstract_level_dict.items():
                    abstract_level_agent_set_dict[abstract_level].add(source)
                    for target,val in source_dict.items():
                        abstract_level_agent_set_dict[abstract_level].add(target)
                        
    abstract_level_idx_dict = {abs_lev:{} for abs_lev in abstract_level_agent_set_dict.keys()}
    
    for abs_lev,agent_set in abstract_level_agent_set_dict.items():
        agent_list = list(agent_set)
        agent_list.sort()
        
        abstract_level_idx_dict[abs_lev] = {agent:idx for idx,agent in enumerate(agent_list)}
                    
    return abstract_level_idx_dict
                    
                    
    
    
    
# will be non-truncated
def GetSectorDict(df,agent_set,country_faction_set_dict):
    d = {}
    for row in range(df.shape[0]):
        # get truncated values
        source_agent_sectors = df['Source Sectors'][row].split(',')
        source_country = df['Source Country'][row]
        source_agent_sectors = [source_country+"::"+source_agent for source_agent in source_agent_sectors]
        country_faction_set_dict.setdefault(source_country,set())
        target_agent_sectors = df['Target Sectors'][row].split(',')
        target_country = df['Target Country'][row]
        country_faction_set_dict.setdefault(target_country,set())
        target_agent_sectors = [target_country+"::"+target_agent for target_agent in target_agent_sectors]
        
        val = df['Intensity'][row]
        
        for source_agent in source_agent_sectors:
            agent_set.add(source_agent)
            country_faction_set_dict[source_country].add(source_agent)
            country_faction_set_dict[target_country].add(source_agent)
        for target_agent in target_agent_sectors:
            agent_set.add(target_agent)
            country_faction_set_dict[source_country].add(target_agent)
            country_faction_set_dict[target_country].add(target_agent)
      
         
        # for now, we will just aggregate the values
        for source_agent in source_agent_sectors:
            for target_agent in target_agent_sectors:
                d.setdefault(source_agent,{})
                d[source_agent].setdefault(target_agent,0.0)
                d[source_agent][target_agent] += val
        
    return d

def GetAdjMatGraphDict(event_dict,agent_idx_dict,num_time_steps):
    
    adjmatgraph_dict = {agent:np.zeros((num_time_steps,
                                                      len(actor_idx_dict.keys()),
                                                      len(actor_idx_dict.keys())))
                        for agent,actor_idx_dict in agent_idx_dict.items()}
    idx=0
    for year,year_dict in event_dict.items():
        print(year)
        for month,temp_event_dict in year_dict.items():
            for agent in agent_idx_dict.keys():
                #print(agent,year,month)
                if agent in temp_event_dict.keys():
                    agent_event_dict = temp_event_dict[agent] 
                    temp_adj_mat = GetAdjMatFromTempEventDict(agent_event_dict,
                                                             agent_idx_dict[agent])
                    adjmatgraph_dict[agent][idx,:,:] = temp_adj_mat
                    del temp_event_dict[agent]
                    #print('temp_event_dict',len(temp_event_dict))
                
                else:
                    # there will be zeros in the adjacency matrix
                    #print('no events for', agent)
                    pass
            idx+=1
            
    return adjmatgraph_dict


def GetAdjMatFromTempEventDict(d,agent_idx_dict):
    
    mat = np.zeros((len(agent_idx_dict.keys()),len(agent_idx_dict.keys())))
    
    for source_name,source_data in d.items():
        source_idx = agent_idx_dict[source_name]
        for target_name,intensity in source_data.items():
            target_idx = agent_idx_dict[target_name]
            mat[source_idx,target_idx] += intensity # is only set once
    
    return ScaleDeleteDiagonal(mat)



def ScaleDeleteDiagonal(mat):
    
    eye_off_set = -(np.eye(mat.shape[0]) - 1)
    mat = np.multiply(mat,eye_off_set)
    
    #Smat = mat / abs(mat).max()
    
    return mat

def GetGraphFromAdjMat(mat,agent_idx_dict,nmc_dict):
    # reverse the keys,val in the idx_dict
    idx_agent_dict = {val:key for key,val in agent_idx_dict.items()}
    
    # scale the graph
    mat = mat/abs(mat).sum()
    
    G = nx.DiGraph()
    
    pop = GetPopArr(mat)#GetNMCPopArr(idx_agent_dict,nmc_dict)

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
        

def GetPopArr(mat):
    abs_mat = abs(mat)
    pop_mat = mat.sum(0) / abs_mat.max()
    #print(pop_mat.shape)
    return pop_mat

