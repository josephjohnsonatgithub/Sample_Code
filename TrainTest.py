import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from easyesn import OneHotEncoder
#from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from scipy.spatial.distance import squareform
from collections import defaultdict
from scipy.cluster import hierarchy
from copy import deepcopy
import nolds
import matplotlib.pyplot as plt
import time

from easyesn import PredictionESN


from pyradox_tabular.model_config import NeuralObliviousDecisionEnsembleConfig
from pyradox_tabular.nn import NeuralObliviousDecisionEnsemble
from pyradox_tabular.model_config import FeatureTokenizerTransformerConfig
from pyradox_tabular.nn import FeatureTokenizerTransformer
from pyradox_tabular.data import DataLoader
from pyradox_tabular.data_config import DataConfig


import importlib
import AugmentData as agdta
importlib.reload(agdta)


def SplitData(df_in,year):
   
    train_df = df_in[df_in['year'] < year]
    test_df = df_in[df_in['year'] >= year]
    
    train_df.reset_index(inplace=True,drop=True)
    
    test_df.reset_index(inplace=True,drop=True)
  
    
    return train_df,test_df

def SegmentResults(feature_df,y_df):
    
    train_X_df,test_X_df = SplitData(feature_df,2012)
    train_y_df,test_y_df = SplitData(y_df,2012)
    
    feature_list = list(feature_df.columns)
    label_list = list(y_df.columns)
    
    feature_list.pop(feature_list.index('year'))
    label_list.pop(label_list.index('year'))
    
    agg_S_slice = set(range(6))
    faction_S_slice = set(range(6,13+6))
    node_S_slice = set(range(13+6,21+6))
    graph_S_slice = set(range(21+6,34+6))
    
    S_slice = set(range(34+6))
    MD_slice = set(range(34+6,37+6))
    D_slice = set(range(37+6,71+12))
    
    agg_D_slice = set(range(37+6,43+6))
    faction_D_slice = set(range(43+6,50+12))
    node_D_slice = set(range(50+12,58+12))
    graph_D_slice = set(range(58+12,71+12))
    
    seg_dict = {}
    
    seg_dict['agg'] = {}
    seg_dict['agg']['S'] = agg_S_slice
    seg_dict['agg']['D'] = agg_D_slice
    seg_dict['agg']['D&MD'] = agg_D_slice.union(MD_slice)
    seg_dict['agg']['ALL'] = agg_S_slice.union(MD_slice).union(agg_D_slice)
    
    seg_dict['faction'] = {}
    seg_dict['faction']['S'] = faction_S_slice
    seg_dict['faction']['D'] = faction_D_slice
    seg_dict['faction']['D&MD'] = faction_D_slice.union(MD_slice)
    seg_dict['faction']['ALL'] = faction_S_slice.union(MD_slice).union(faction_D_slice)
    
    seg_dict['node'] = {}
    seg_dict['node']['S'] = node_S_slice
    seg_dict['node']['D'] = node_D_slice
    seg_dict['node']['D&MD'] = node_D_slice.union(MD_slice)
    seg_dict['node']['ALL'] = node_S_slice.union(MD_slice).union(node_D_slice)
    
    seg_dict['graph'] = {}
    seg_dict['graph']['S'] = graph_S_slice
    seg_dict['graph']['D'] = graph_D_slice
    seg_dict['graph']['D&MD'] = graph_D_slice.union(MD_slice)
    seg_dict['graph']['ALL'] = graph_S_slice.union(MD_slice).union(graph_D_slice)
    
    seg_list = list(seg_dict.keys())
    
    best_score_dict = {cat:{'best':0.0,'sub-groups':[]} for cat in ['dpc','erv','ic','ins','reb','ilc']}
    
    for idx,key1 in enumerate(seg_list):
        for key2 in seg_list[idx+1:]:
            val1 = seg_dict[key1]
            val2 = seg_dict[key2]
            print('###################')
            print()
            for data_type1,type_slice1 in val1.items():
                for data_type2,type_slice2 in val2.items():
                    print('\n',key1,data_type1,'\t',key2,data_type2)
                    feature_vec = np.array(feature_list)
                    selected_feature_vec = feature_vec[list(type_slice1.union(type_slice2))]
                    X_train = train_X_df[list(selected_feature_vec)]
                    X_test = test_X_df[list(selected_feature_vec)]
            
                    for label in ['dpc','erv','ic','ins','reb','ilc']:
                        y_train = train_y_df[label].ravel()
                        y_test = test_y_df[label].ravel()
                        clf = BaggingClassifier(base_estimator=RandomForestClassifier(n_estimators=15),n_estimators=10).fit(X_train,y_train)
                        temp_auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])
                        if temp_auc > best_score_dict[label]['best']:
                            best_score_dict[label]['best'] = temp_auc
                            best_score_dict[label]['sub-groups'] = [key1,data_type1,key2,data_type2]
                        
                        print(label,round(roc_auc_score(y_test, clf.predict_proba(X_test)[:,1]),3))
                    for k,v in best_score_dict.items():
                        print(k)
                        print(round(v['best'],3))
                        print(v['sub-groups'])
            
            
    
def GetResults(feature_df,y_df):
    
    train_X_df,test_X_df = SplitData(feature_df,2012)
    train_y_df,test_y_df = SplitData(y_df,2012)
    
    
    feature_list = list(feature_df.columns)
    label_list = list(y_df.columns)
    
    feature_list.pop(feature_list.index('year'))
    label_list.pop(label_list.index('year'))
    
    selected_features = feature_list
    
    scaler = MinMaxScaler()
    X_train = train_X_df[selected_features]
    #X_train = scaler.fit_transform(X_train)
    X_test = test_X_df[selected_features]
    #X_test = scaler.transform(X_test)
    
    for label in ['dpc','erv','ic','ins','reb','ilc']:
        
        y_train = train_y_df[label].ravel()
        y_test = test_y_df[label].ravel()
        clf = RandomForestClassifier(n_estimators=750).fit(X_train,y_train)
        #clf = BaggingClassifier(base_estimator=RandomForestClassifier(n_estimators=15),n_estimators=10).fit(X_train,y_train)
        print(label,round(roc_auc_score(y_test, clf.predict_proba(X_test)[:,1]),3))
        
def GetSegResults(feature_df,y_df):
    
    scaler = MinMaxScaler()
    X_train = train_X_df[selected_features]
    #X_train = scaler.fit_transform(X_train)
    X_test = test_X_df[selected_features]
    #X_test = scaler.transform(X_test)
    
    for label in ['dpc','erv','ic','ins','reb','ilc']:
        
        y_train = train_y_df[label].ravel()
        y_test = test_y_df[label].ravel()
        #clf = RandomForestClassifier(n_estimators=500).fit(X_train,y_train)
        clf = BaggingClassifier(base_estimator=RandomForestClassifier(n_estimators=15),n_estimators=10).fit(X_train,y_train)
        print(label,round(roc_auc_score(y_test, clf.predict_proba(X_test)[:,1]),3))
        
        
def TestClassLabels(agent_X_dict,agent_y_dict,T_in,w,offset,aug_T,inf_d,test_type):
    
    description_str = "ALL_"+str(offset+1)+"_"+str(w) + "_" + str(T_in) + "_" + str(aug_T) + "_" + test_type#_sim_" + "_correct"
    #in ['Bahamas','Barbados','Belize','Cape Verde','Comoros','Equatorial Guinea','Iceland']:
    #trn = 12
    cls = 48
    start = time.time()
    
    for agent,vec in agent_X_dict.items():
        
        print(agent)
        temp_feat_arr = vec
        
        T = T_in #int(chaos_dict[agent]*T_in)
        #print(T)
        
        y = agent_y_dict[agent][offset+1:]
        # drop last row
        X = np.array(temp_feat_arr[36:-1-offset])
        print(y.shape,X.shape)
        
        #ohe = OneHotEncoder(handle_unknown='ignore')
        #ohe.fit(y.reshape(-1,1))
        #y_n = ohe.transform(y.reshape(-1,1)).toarray()        
        tscv = TimeSeriesSplit(n_splits=len(X)-1)
        
        y_last_test = ''
        num_correct_naive = 0
        num_correct_majority=0
        num_correct = 0
        num = 0
        y_latest =''
        n_est=10
        
        for idx,(train_index, test_index) in enumerate(tscv.split(X)):
        
            if idx < (cls-1):
                continue
               
            temp_train_index = train_index[max(0,idx-T-offset+1):] # idx-T-offset+1
                
            y_train, y_test = y[temp_train_index], y[test_index][0]
            X_train, X_test = X[temp_train_index], X[test_index]
            #X_train_label, X_test_label = y_n[temp_train_index], y_n[test_index]
            #X_train_method, X_test_method = X[temp_train_index], X[test_index]
            
            #X_train = np.random.rand(X_train.shape[0],X_train.shape[1])
            #X_test = np.random.rand(X_test.shape[1]).reshape(1,-1)
            
            #X_train = X_train_label[:-1]#.reshape(-1,1)
            #X_train_method = X_train_method[1:]
            #X_test = X_train_label[-1].reshape(1,-1)
            #y_train = y_train[1:]#.reshape(-1,1)
            #X_train = np.concatenate((X_train_method,X_train_label),axis=1)
            #X_test = np.concatenate((X_test_method,X_test_label),axis=1)
            
            #scaler = MinMaxScaler()
            #X_train = scaler.fit_transform(X_train)
            #X_test = scaler.transform(X_test)
            
            
            
            X_train,X_test = TrimX(X_train,X_test)
            
            
            '''
            corr = spearmanr(X_train).correlation

            # Ensure the correlation matrix is symmetric
            corr = (corr + corr.T) / 2
            np.fill_diagonal(corr, 1)

            # We convert the correlation matrix to a distance matrix before performing
            # hierarchical clustering using Ward's linkage.
            distance_matrix = 1 - np.abs(corr)
            
            try:
                dist_linkage = hierarchy.ward(squareform(distance_matrix))
                cluster_ids = hierarchy.fcluster(dist_linkage, 0.5, criterion="distance")
                cluster_id_to_feature_ids = defaultdict(list)
                for idx, cluster_id in enumerate(cluster_ids):
                    cluster_id_to_feature_ids[cluster_id].append(idx)
                selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]
            except:
                selected_features = list(np.arange(36))

            X_train = X_train[:, selected_features]
            X_test = X_test[:,selected_features]'''
            
            y_latest = y_train[-1]
            
            #ssq_vec = ((X_train - X_test)**2).sum(1)
            #sim_weights = abs(ssq_vec-ssq_vec.mean())
            #sim_weights = sim_weights / sim_weights.max()
            #time_weights = np.arange(100,len(y_train)+100)
            #time_weights = time_weights / time_weights.max()
            #weights = sim_weights*time_weights
            
            clf = BaggingClassifier(base_estimator=RandomForestClassifier())
            clf.fit(X_train,y_train)
            
            #clf = RandomForestClassifier(10).fit(X_train,y_train)
            #clf = KNeighborsClassifier(3).fit(X_train,y_train)
            #clf = ExtraTreesClassifier(10).fit(X_train,y_train)
            
            pred = clf.predict(X_test)[0]
            
            
            num_correct += pred == y_test
            num_correct_naive += y_latest == y_test
            num_correct_majority+=MajorityLearner(y_train)==y_test
            num += 1
        print('num_preds:\t\t',num)
        print('accuracy:\t',num_correct/num)
        print('naive:\t\t',num_correct_naive/num,num)
        print('majority:\t',num_correct_majority/num,num)
        end = time.time()
        print(end - start)
        print()

        start = time.time()

        f = open('icews_quad_accuracy.csv','a') # include nan countries
        f.write(agent + "," + str(num_correct/num) + "," 
                    + str(num_correct_naive/num) + "," 
                    + str(num_correct_majority/num) + "," 
                    + description_str + "\n")
        f.close()
        
def TrimX(X_train,X_test):
    
    col_rep = abs(X_train.mean(0) - X_train.max(0)) < .00001
    
    delete_col_rep = [i for i,x in enumerate(col_rep) if x == True]
    
    delete_idx_list = delete_col_rep
    
    X_train = np.delete(X_train,delete_idx_list,1)
    X_test = np.delete(X_test,delete_idx_list,1)
    
    return X_train,X_test
    
    
        
def MajorityLearner(vec):
    label_list = list(set(vec))
    hit_list = []
    
    for label in label_list:
        hit_list.append((np.array(vec)==label).sum())
        
    return label_list[np.array(hit_list).argmax()]
    
        
def GetQuartileMultiple(y_agg_vals):
    # assigns quartiles to each value
    first = np.percentile(y_agg_vals,25)
    second = np.percentile(y_agg_vals,50)
    third = np.percentile(y_agg_vals,75)
    y_cat_arr = np.full(len(y_agg_vals),'fourth')
    
    y_cat_arr[y_agg_vals <= third] = 'third '
    y_cat_arr[y_agg_vals <= second] = 'second'
    y_cat_arr[y_agg_vals <= first] = 'first '
    
    return y_cat_arr,first,second,third 

def GetQuartileSingle(y_val,first,second,third):
    if y_val <= first:
        return 'first '
    elif y_val <= second:
        return 'second'
    elif y_val <= third:
        return 'third '
    else:
        return 'fourth'
    
    
def TestClassLabelsAUC(agent_X_dict,agent_y_dict,T_in,w,offset,aug_T,inf_d,test_type):
    
    description_str = "ALL_"+str(offset+1)+"_"+str(w) + "_" + str(T_in) + "_" + str(aug_T) + "_" + test_type#_sim_" + "_correct"
    
    #trn = 12
    cls = 48
    
    for agent,vec in agent_X_dict.items():
        if agent in ['Bahamas','Barbados',
                     'Belize','Cape Verde',
                     'Comoros','Equatorial Guinea',
                     'Iceland','Luxembourg','Sao Tome and Principe',
                    'Seychelles']:
            continue
        
        print(agent)
        temp_feat_arr = vec
        
        T = T_in #int(chaos_dict[agent]*T_in)
        #print(T)
        
        y = agent_y_dict[agent][offset+1:]
        # drop last row
        X = np.array(temp_feat_arr[36:-1-offset])
        print(y.shape,X.shape)
        
        tscv = TimeSeriesSplit(n_splits=len(X)-1)
        
        naive_preds=[]
        majority_preds=[]
        method_preds=[]
        actual_y=[]
        
        for idx,(train_index, test_index) in enumerate(tscv.split(X)):
        
            if idx < (cls-1):
                continue
               
            temp_train_index = train_index[max(0,idx-T+1-offset):]
                
            y_train, y_test = y[temp_train_index], y[test_index]
            X_train, X_test = X[temp_train_index], X[test_index]
            
            X_train,X_test = TrimX(X_train,X_test)
            
            '''corr = spearmanr(X_train).correlation

            # Ensure the correlation matrix is symmetric
            corr = (corr + corr.T) / 2
            np.fill_diagonal(corr, 1)

            # We convert the correlation matrix to a distance matrix before performing
            # hierarchical clustering using Ward's linkage.
            distance_matrix = 1 - np.abs(corr)
            try:
                dist_linkage = hierarchy.ward(squareform(distance_matrix))
                cluster_ids = hierarchy.fcluster(dist_linkage, 0.5, criterion="distance")
                cluster_id_to_feature_ids = defaultdict(list)
                for idx, cluster_id in enumerate(cluster_ids):
                    cluster_id_to_feature_ids[cluster_id].append(idx)
                selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]
            except:
                selected_features = list(np.arange(13))

            X_train = X_train[:, selected_features]
            X_test = X_test[:,selected_features]
            
            y_latest = y_train[-1]
            
            ssq_vec = ((X_train - X_test)**2).sum(1)
            sim_weights = ssq_vec #abs(ssq_vec-ssq_vec.max())
            sim_weights = sim_weights / sim_weights.max()
            time_weights = np.arange(1,len(y_train)+1)
            time_weights = time_weights / time_weights.max()
            weights = sim_weights*time_weights'''
            
            clf = BaggingClassifier(base_estimator=RandomForestClassifier(n_estimators=15),n_estimators=10).fit(
                X_train,
                y_train)
            
            pred = clf.predict_proba(X_test)
            actual_y.append(y_test.tolist()[0])
            method_preds.append(pred.tolist()[0][1])
            naive_preds.append(y_train[-1])
            
            if y_train.mean()>.5:
                majority_preds.append(1)
            else:
                majority_preds.append(0)
            #print(actual_y)
            #print(method_preds)
            #print(naive_preds)
            #print(majority_preds)
            
            
        print(offset,aug_T)
        print('accuracy:\t',roc_auc_score(actual_y, method_preds))
        print('naive:\t\t',roc_auc_score(actual_y, naive_preds))
        print('majority:\t\t',roc_auc_score(actual_y, majority_preds))
        print()
        
        f = open('icews_auc.csv','a') # include nan countries
        f.write(agent + "," + str(round(roc_auc_score(actual_y, method_preds),4)) + "," 
                + str(round(roc_auc_score(actual_y, naive_preds),4)) + "," 
                + str(round(roc_auc_score(actual_y, majority_preds),4)) + "," 
                + description_str + "\n")
        f.close()
        
def NoRegret(agent_X_dict,agent_y_dict,T_in,w,offset,aug_T,inf_d,test_type,chaos_dict):
    
    description_str = "ALL_"+str(offset+1)+"_"+str(w) + "_" + str(T_in) + "_" + str(aug_T) + "_" + test_type#_sim_" + "_correct"
    
    #trn = 12
    cls = 48
    start = time.time()
    for agent,vec in agent_X_dict.items():
        
        print(agent)
        temp_feat_arr = vec
        
        T = T_in #int(chaos_dict[agent]*T_in)
        #print(T)
        
        y = agent_y_dict[agent][offset+1:]
        # drop last row
        X = np.array(temp_feat_arr[36:-1-offset])
        print(y.shape,X.shape)
        
        tscv = TimeSeriesSplit(n_splits=len(X)-1)
        
        y_last_test = ''
        num_correct_naive = 0
        num_correct_majority=0
        num_correct = 0
        num = 0
        y_latest =''
        for idx,(train_index, test_index) in enumerate(tscv.split(X)):
        
            if idx < (cls-1):
                continue
               
            temp_train_index = train_index[max(0,idx-T+1-offset):]
            temp_nn_train_index = train_index[:max(0,idx-T+1-offset)]
            
            if offset > 0:
                temp_train_index = temp_train_index[:-offset]
                
            y_train, y_test = y[temp_train_index], y[test_index][0]
            X_train, X_test = X[temp_train_index], X[test_index]
            
            
            y_latest = y_train[-1]
            
            if NaiveAccuracy(y_train) >= MajorityAccuracy(y_train):
                pred = y_train[-1]
            else:
                pred = MajorityLearner(y_train)
            
            num_correct += pred == y_test
            num_correct_naive += y_latest == y_test
            num_correct_majority+=MajorityLearner(y_train)==y_test
            num += 1
            
        print('accuracy:\t',round(num_correct/num,3))
        print('naive:\t\t',round(num_correct_naive/num,3),num)
        print('majority:\t',round(num_correct_majority/num,3),num)
        end = time.time()
        print(end - start)
        print()
        
        start = time.time()
        
        f = open('icews_quad_accuracy.csv','a') # include nan countries
        f.write(agent + "," + str(round(num_correct/num,3)) + "," 
                + str(round(num_correct_naive/num,3)) + "," 
                + str(round(num_correct_majority/num,3)) + "," 
                + description_str + "\n")
        f.close()
        
def MajorityAccuracy(y_train):
    # predict one step ahead
    num_correct = 0
    for idx,label in enumerate(y_train[:-1]):
        maj_pred = MajorityLearner(y_train[:idx+1])
        if maj_pred==label:
            num_correct+=1
    return num_correct/len(y_train[:-1])
        
def NaiveAccuracy(y_train):
    return (y_train[1:]==y_train[:-1]).sum() / len(y_train[:-1])


def TestClassLabelsNODE(agent_X_dict,agent_y_dict,T_in,w,offset,aug_T,inf_d,test_type):
    
    description_str = "ALL_"+str(offset+1)+"_"+str(w) + "_" + str(T_in) + "_" + str(aug_T) + "_" + test_type#_sim_" + "_correct"
    #in ['Bahamas','Barbados','Belize','Cape Verde','Comoros','Equatorial Guinea','Iceland']:
    #trn = 12
    cls = 48
    start = time.time()
    for agent,vec in agent_X_dict.items():
        
        print(agent)
        temp_feat_arr = vec
        
        T = T_in #int(chaos_dict[agent]*T_in)
        #print(T)
        
        y = agent_y_dict[agent][offset+1:]
        ohe = OneHotEncoder()
        y = ohe.fit_transform(y)
        # drop last row
        X = np.array(temp_feat_arr[36:-1-offset])
        print(y.shape,X.shape)
        
              
        tscv = TimeSeriesSplit(n_splits=len(X)-1)
        
        y_last_test = ''
        num_correct_naive = 0
        num_correct_majority=0
        num_correct = 0
        num = 0
        y_latest =''
        for idx,(train_index, test_index) in enumerate(tscv.split(X)):
        
            if idx < (cls-1):
                continue
               
            temp_train_index = train_index[max(0,idx-T-offset+1):] # idx-T-offset+1
            
            
            y_train, y_test = y[temp_train_index], y[test_index]
            X_train, X_test = X[temp_train_index], X[test_index]
            
            print(X_train.shape,y_train.shape)
            X_train,X_test = TrimX(X_train,X_test)
            
            num_feats = X_train.shape[1]
            num_classes = y_train.shape[1]
            
            esn = PredictionESN(num_feats,100,num_classes)
            
            esn.fit(X_train, y_train, transientTime=0, verbose=0)
            
            pred = esn.predict(X_test)
            
            y_latest = y_train[-1]
            
            
            num_correct += pred.argmax() == y_test.argmax()
            num_correct_naive += y_latest.argmax() == y_test.argmax()
            
            num += 1
            print('num_preds:\t\t',num)
            print('accuracy:\t',round(num_correct/num,3))
            print('naive:\t\t',round(num_correct_naive/num,3),num)
            
        end = time.time()
        print(end - start)
        print()
        
        start = time.time()
        continue
        f = open('icews_quad_accuracy.csv','a') # include nan countries
        f.write(agent + "," + str(round(num_correct/num,3)) + "," 
                + str(round(num_correct_naive/num,3)) + "," 
                + str(round(num_correct_majority/num,3)) + "," 
                + description_str + "\n")
        f.close()