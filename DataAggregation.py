# aggregates the data in different ways

import pandas as pd
import numpy as np
from copy import deepcopy


# number of events by quad by country by [source,target]
def AggregateQuadData(year_range,df,agent_list,abb_size):
    
    quad_feature_list = ["QuadClass1","QuadClass2","QuadClass3","QuadClass4"]
    source_dict = InstantiateDict(year_range,agent_list,quad_feature_list)
    target_dict = InstantiateDict(year_range,agent_list,quad_feature_list)
    
    for row in range(17000000,df.shape[0]):
        
        t_year = df['Date'][row].year
        if row%1000000==0:
            print(row,t_year)
        
        if t_year not in year_range:
            continue

        t_source_abb = df['Source'][row][:abb_size]
        t_target_abb = df['Target'][row][:abb_size]
        t_num_events = df['NumEvents'][row]
        t_year = df['Date'][row].year
        t_month = df['Date'][row].month
        t_quad_class = "QuadClass" + str(df['QuadClass'][row])
        
        
        if t_source_abb in agent_list:
            
            source_dict[t_year][t_month][t_source_abb][t_quad_class] += t_num_events
        if t_target_abb in agent_list:
            
            target_dict[t_year][t_month][t_target_abb][t_quad_class] += t_num_events
            
    return source_dict,target_dict
    
def InstantiateDict(year_range,agent_list,feature_list):
    
    temp_dict = {}
    feature_dict = {feature:0 for feature in feature_list}
    
    for year in year_range:
        temp_dict.setdefault(year,{})
        for month in range(1,13):
            temp_dict[year].setdefault(month,{})
            for agent in agent_list:
                temp_dict[year][month].setdefault(agent,deepcopy(feature_dict))
    
    return temp_dict

def AggregateWGIData(year_range,country_agent_list):
    
    df, key_indicator_list = GetIndicatorData()
    
    indicator_year_month_dict = InstantiateDict(year_range,country_agent_list,key_indicator_list)

    for year in year_range:
        for month in range(1,13):
            print(year,month)
            if str(year) not in df.columns:
                continue
            for country_abb in country_agent_list:
                try:
                    temp_dict = {t_ind:GetIndVal(df,t_ind,country_abb,year-1) for t_ind in key_indicator_list}
                    indicator_year_month_dict[year][month][country_abb] = temp_dict
                except:
                    print(country_abb)
    return indicator_year_month_dict
    
            
def GetIndicatorData():
    df = pd.read_csv('GDELTData/WGIData.csv')
    full_indicator_list = list(set(df['Indicator Name']))
    key_indicator_list = []
    for ind in full_indicator_list:
        if "Estimate" in ind:
            key_indicator_list.append(ind)
    return df,key_indicator_list

def GetIndVal(df,indicator_name,country_abb,year):
    series_tok = df[(df['Indicator Name']==indicator_name) & (df['Country Name']==country_abb)][str(year)]
    return series_tok.to_numpy()[0]
                
            
            
    
    
        
            
            