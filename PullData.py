import pandas as pd
import numpy as np
import datetime
import os
import networkx as nx
import nolds
from math import log
from copy import deepcopy

# anytime dfs need to be accessed, we will use this module

# sweep through and get all events (map events to countries)
# get event dict based on events


def GetDfDictByYear(year_range):
    
    df_dict = {year:None for year in year_range}
    name_set=set()
    
    tab_docs = os.listdir("EventTables")
    
    for year in df_dict.keys():
        
        doc = GetDoc(year,tab_docs)
        temp_df = pd.read_table("EventTables/" + doc)
        print('downloaded:',year,temp_df.shape)
        
        temp_df['Event Date'] = pd.to_datetime(temp_df['Event Date'],format='%Y-%m-%d')
        print('date parsed')
        
        # clean up names for standardization
        temp_df['Source Name'] = temp_df['Source Name'].str.strip()
        temp_df['Source Sectors'] = temp_df['Source Sectors'].str.strip()
        temp_df['Source Country'] = temp_df['Source Country'].str.strip()
        
        temp_df['Target Name'] = temp_df['Target Name'].str.strip()
        temp_df['Target Sectors'] = temp_df['Target Sectors'].str.strip()
        temp_df['Target Country'] = temp_df['Target Country'].str.strip()
        
        temp_df['Source Sectors'] = temp_df['Source Sectors'].fillna('*')
        temp_df['Target Sectors'] = temp_df['Target Sectors'].fillna('*')
        
        #temp_df['Source Country'] = temp_df['Source Country'].fillna('NGO')
        #temp_df['Target Country'] = temp_df['Target Country'].fillna('NGO')
        
        
        
        print('agents cleaned')
        temp_df.drop(temp_df.columns[list(range(12,20))], axis=1, inplace=True)
        print('columns dropped')
        
        # drop id columns
        temp_df.drop(['Event ID', 'Story ID'], axis=1, inplace=True)
        #drop duplicates
        temp_df.drop_duplicates(inplace=True)
        temp_df.reset_index(inplace=True,drop=True)
        
        # country actors
        NanToActor(temp_df,name_set)
        print('before dropna',temp_df.shape)
        temp_df.dropna(inplace=True)
        print('after dropna',temp_df.shape)
        temp_df.reset_index(inplace=True,drop=True)
        
        print(temp_df.shape)
        
        print(len(set(temp_df['Source Country'])))
        
        df_dict[year] = temp_df
    return df_dict

def NanToActor(df,name_set):
    
    for row in range(df.shape[0]):
        if pd.isna(df['Source Country'][row]):
            name = df['Source Name'][row]
            if "(" in name and ")" in name:
                text = name.split("(")[1]
                text = text.split(")")[0]
                df.at[row,"Source Country"]=text
            else:
                df.at[row,"Source Country"]="NSA"
        if pd.isna(df['Target Country'][row]):
            name = df['Target Name'][row]
            if "(" in name and ")" in name:
                text = name.split("(")[1]
                text = text.split(")")[0]
                df.at[row,"Target Country"]=text
            else:
                df.at[row,"Target Country"]="NSA"
    

def GetCountryData():
    
    df = pd.read_csv('Data/gtds_2001.to.may.2014.csv',header=0)
    
    country_predict_list = list(set(df['country']))
    country_predict_list.sort()
    
    country_region_dict = {}
    region_country_set_dict = {}
    for row in range(df.shape[0]):
        temp_country = df['country'][row]
        temp_region = df['region'][row]
        country_region_dict.setdefault(temp_country,temp_region)
        region_country_set_dict.setdefault(temp_region,set())
        region_country_set_dict[temp_region].add(temp_country)
    
    return country_predict_list, country_region_dict, region_country_set_dict

# get labeled data for each country
def GetViolQuadDataCountry(year_df_dict,country_predict_list):
    
    country_violence_dict = {country:{year:{month:0.0 for month in range(1,13)} for year in year_df_dict.keys()} for country in country_predict_list}
    
    for year,df in year_df_dict.items():
        
        temp_df = df[df['Intensity'] <= -7.0]
        temp_df.reset_index(inplace=True,drop=True)
        print(year,temp_df['Intensity'].sum())
        
        for row in range(temp_df.shape[0]):
            
            val = -temp_df['Intensity'][row]
            month = temp_df['Event Date'][row].month
            src_country = temp_df['Source Country'][row]
            tgt_country = temp_df['Target Country'][row]
            if src_country in country_predict_list:
                country_violence_dict[src_country][year][month] += val
            if (tgt_country in country_predict_list) and (tgt_country != src_country):
                country_violence_dict[tgt_country][year][month] += val
                
    return country_violence_dict

# get labeled data for each region
def GetViolQuadDataRegion(year_df_dict,country_region_dict):
    
    region_list = list(set(country_region_dict.values()))
    print(region_list)
    
    region_violence_dict = {region:{year:{month:0.0 for month in range(1,13)} for year in year_df_dict.keys()} for region in region_list}
    
    for year,df in year_df_dict.items():
        
        temp_df = df[df['Intensity'] <= -7.0]
        temp_df.reset_index(inplace=True,drop=True)
        print(year,temp_df['Intensity'].sum())
        
        for row in range(temp_df.shape[0]):
            
            val = -temp_df['Intensity'][row]
            month = temp_df['Event Date'][row].month
            src_country = temp_df['Source Country'][row]
            tgt_country = temp_df['Target Country'][row]
            if src_country in country_region_dict.keys():
                region_violence_dict[country_region_dict[src_country]][year][month] += val
            if (tgt_country in country_region_dict.keys()):
                if (src_country not in country_region_dict.keys()):
                    region_violence_dict[country_region_dict[tgt_country]][year][month] += val
                elif (country_region_dict[tgt_country] != country_region_dict[src_country]):
                    region_violence_dict[country_region_dict[tgt_country]][year][month] += val
                
    return region_violence_dict

# this dict will only have the countries that we will be predicting on as keys
def GetClassVec(country_violence_dict,num_time_steps):
    
    class_labels_dict = {country:None for country in country_violence_dict}
    chaos_dict = {country:None for country in country_violence_dict}
    
    
    for country,year_month_dict in country_violence_dict.items():
        agg_vec = GetLevelList(year_month_dict,num_time_steps)
        agg_vec = agg_vec[36+12:]
        #hurst_val = nolds.hurst_rs(agg_vec)
        #chaos_dict[country] = hurst_val
        #lyap_val = nolds.lyap_r(agg_vec,emb_dim=12)
        #f = open('hurst_vals.csv','a')
        #f.write(country + "," + str(hurst_val) + "\n")
        #f.close()
        #f = open('lyap_vals.csv','a')
        #f.write(country + "," + str(lyap_val) + "\n")
        #f.close()
        first,second,third,y_cat_arr = GetQuartiles(agg_vec)
        #y_cat_arr = BinaryVec(y_cat_arr)
        class_labels_dict[country] = y_cat_arr
    return class_labels_dict,chaos_dict

def TriadLabels(arr):
    change_arr = deepcopy(arr)
    change_arr[:]='same'
    label_list = ['first ','second','third ','fourth']
    
    for i,label in enumerate(arr):
        if i==0:
            continue
        elif label_list.index(arr[i]) > label_list.index(arr[i-1]):
            change_arr[i]='increase'
        elif label_list.index(arr[i]) < label_list.index(arr[i-1]):
            change_arr[i]='decrease'
        else:
            pass
    return change_arr

def BinaryLabels(arr):
    change_arr = deepcopy(arr)
    change_arr[0]='same'
    for i,j in enumerate(arr):
        if i==0:
            continue
        else:
            if arr[i]==arr[i-1]:
                change_arr[i]="same"
            else:
                change_arr[i]="change"
            
    return change_arr

def BinaryVec(arr):
    change_arr = np.zeros(len(arr))
    for i,j in enumerate(arr):
        if i==0:
            continue
        else:
            if arr[i]==arr[i-1]:
                change_arr[i]=1
            else:
                pass
            
    return change_arr
    
    
# extracts levels of violence from the year_month dict and returns a sequential list
def GetLevelList(year_month_dict,num_time_steps):
    
    agg_vec = np.zeros(num_time_steps)
    i=0
    for year,data1 in year_month_dict.items():
        for month,level in data1.items():
            agg_vec[i] = level
            i+=1
    return agg_vec


def GetQuartiles(y_agg_vals):
    # assigns quartiles to each value
    first = np.percentile(y_agg_vals,25)
    second = np.percentile(y_agg_vals,50)
    third = np.percentile(y_agg_vals,75)
    #print('first',first)
    #print('second',second)
    #print('third',third)
    
    y_cat_arr = np.full(len(y_agg_vals),'fourth')
    y_cat_arr[y_agg_vals < third] = 'third '
    y_cat_arr[y_agg_vals < second] = 'second '
    y_cat_arr[y_agg_vals < first] = 'first '
        
    #print((y_cat_arr=='first ').sum())
    #print((y_cat_arr=='second').sum())
    #print((y_cat_arr=='third ').sum())
    #print((y_cat_arr=='fourth').sum())
    
    return first,second,third,y_cat_arr

def GetDoc(year,tab_docs):
    for doc in tab_docs:
        if "." + str(year) + "." in doc:
            return doc
    
# will be a dict {country:nmc_vec=[]} that mirrors year range
def GetNMC(year_range):
    
    path='NMC-60-abridged.csv'
    
    df = pd.read_csv(path,header=0)
    d={}
    
    for row in range(df.shape[0]):
        
        year=df['year'][row]
        
        if year in year_range:
            country=df['country'][row]
            nmc=df['cinc'][row]
            d.setdefault(year,{})
            d[year][country] = nmc
        
        
    return d
    
    
    
    
    