# performs miscellaneous functions
import numpy as np
import pandas as pd
import os

def CountBlankEntries(node_country_idx_dict):
    
    node_blank_dict = {country:set() for country in node_country_idx_dict.keys()}
    
    for doc in os.listdir("NodeData"):
        
        temp_data = doc.split(".")
        if 'ipynb_checkpoints' in temp_data:
            continue
        year = int(temp_data[0])
        month = int(temp_data[1])
        temp_df = pd.read_csv("NodeData/" + doc,index_col=0,header=0)
        temp_df = temp_df.fillna(0)
        temp_emb_arr = np.array(temp_df)
        
        for country,idx in node_country_idx_dict.items():
            
            if temp_emb_arr[idx,:].sum() == 1.0:
                node_blank_dict[country].add(str(year) + "::" + str(month))
    
    return node_blank_dict


#aug_X,aug_y = agdta.GetRandomData(agent_X_dict,agent_y_dict,temp_train_index,aug_T,X_train.shape[1],trn+offset+1)
            #aug_X,aug_y = agdta.GetViolSimData(agent,agent_X_dict,agent_y_dict,temp_train_index,aug_T,X_train.shape[1],trn+offset+1,inf_d)
            #aug_X,aug_y = agdta.GetRegionSimData(agent,agent_X_dict,agent_y_dict,temp_train_index,aug_T,X_train.shape[1],trn+offset+1,inf_d)
            
            #aug_X,aug_y = agdta.GetRegionMixData(agent,agent_X_dict,agent_y_dict,temp_train_index,aug_T,X_train.shape[1],trn+offset+1,inf_d,chaos_dict)
            
            
            
            
                
            #scaler = MinMaxScaler()
            #X_train = scaler.fit_transform(X_train)
            #X_test = scaler.transform(X_test)
                
            
            
            
            
            
            
        
        