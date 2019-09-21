#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 23:06:12 2017

@author: ambuj
"""
import pandas as pd
import numpy as np
import glob
import itertools
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import Ridge
import time
import random
import math


from scipy.stats.stats import pearsonr
#import sys
#sys.path.insert(0, '/home/ambuj/Documents/Prabakaran/')
#import propylib_28Aug17


method = 'LR'     #  'LR' for Linear regression
                # 'RR' for ridge regression


def divide_seq(seq, bind_inf, size):
    bind_df = pd.DataFrame()
    for i in range(size, len(seq), 1):
        mid = (i-int(size/2))-1
#        print mid
        bind_df = bind_df.append([[seq[i-size:i], bind_inf[mid]]])
    return bind_df

def include_properties(final_mat):
    final_mat.reset_index(inplace=True)
    feature103 = pd.read_csv('features/imp/extracted_feature_new.csv')
    for i in range(len(feature103)):
        colname = feature103.iloc[i]['ID']
        final_mat[colname] ='nan'
#        print feature103.iloc[i]
#    final_mat[['f1'],['f2']] = 'NAN'
    for j in range(len(final_mat)):
        block = final_mat.iloc[j]['blocks']

        l = 0
        for f in feature103['ID']:
            prop_sum = 0


            for k in range(len(block)):
                prop_sum += feature103.iloc[l][block[k:k+1]]

            l += 1
            final_mat.loc[j,f] = prop_sum/(k+1)

    #        print prop_sum

    return final_mat


def linear_regression(df2, results):
    lm = linear_model.LinearRegression()
    model = lm.fit(df2, results)

#    preds = gnb.predict(df2)
    return model

def ridge_regression (df2, results):
    rlm = Ridge(alpha=1.0)
    model = rlm.fit(df2, results)
    return model

def decision_tree(df2, results):
    dt = tree.DecisionTreeClassifier(criterion='gini')
    dt.fit(df2, results)
#    dt.score(df2,results)
#    
#    preds = dt.predict(df2)
    return dt


def support_vector_machine(df2, results):
    svm2 = svm.SVC()
    svm2.fit(df2, results)
    svm2.score(df2, results)

#    preds = svm2.predict(df2)
    return svm2


def random_forest(df2, results):
    rf1 = RandomForestClassifier()

    rf1 = rf1.fit(df2, results)
#    rf1 = SelectFromModel(rf1, prefit=True)
#    
#    new_data = rf1.transform(df2)
##    print new_data.shape[1]
#    rf2 = RandomForestClassifier()
#    rf2.fit(new_data,results)
#    preds = rf2.predict(new_data)
    return rf1


# =============================================================================
# def back_sel_by_accuracy2(df2,method='GNB',validation='NFCV', num_steps=1):
#     
#     hold = 0
#     df4 = df2.drop(['results'],axis = 1)
#     df3 = df2
# #    print num_steps
#     for steps in range(num_steps):
#         
#         for cols in df4.columns:
#             
#             df3 = df3.drop([cols],axis = 1)
# #==============================================================================
# #             if (method == 'GNB'):
# #                 model = gausian_naive_bayes(df3,results)
# #                 preds = model.predict(df3)
# #             elif (method == 'DT'):
# #                 model = decision_tree(df3,results)
# #                 preds = model.predict(df3)
# #             elif (method == 'SVM'):
# #                 model = support_vector_machine(df3,results)
# #                 preds = model.predict(df3)
# #             elif (method == 'RF'):
# #                 model = random_forest(df3,results)
# #                 preds = model.predict(df3)
# #==============================================================================
#                 
# #            print preds
#             if validation == 'NFCV':
#                 senstivity, specificity, accuracy, accuracy2 = nfold_cv(df3,10, method)
#             elif validation == 'JKT':
#                 senstivity, specificity, accuracy, accuracy2 = JK_Test(df3, method)
# #            tp,tn,fp,fn = JK_Test(df3,method)
# #            senstivity = float(tp)/float(tp+fn)
# #            specificity = float(tn)/float(tn+fp)
# ##            accuracy = float(tp+tn)/float(tp+tn+fp+fn)
# #            accuracy2 = float(senstivity+specificity)/2
#             
#             if accuracy2 - hold <= 0:
#                 
#                 df3 = df2
#                 
#             else:
#                 
# #                print cols,hold, accuracy2
#                 hold = accuracy2
#                 df2 = df3
#                 
#             if hold == 0:
#                 hold = accuracy2
#         
#             
#     return df2
# =============================================================================


def frwd_sel_by_accuracy2(df2, method='GNB', validation='NFCV', num_steps=1):
    hold = 0
    df4 = df2.drop(['delG'],axis = 1)
    df3 = pd.DataFrame()
#    df5 = pd.DataFrame()
    for steps in range(num_steps):
        col_list = list(df4.columns)
        for cols in col_list:
            df3[cols] = df2[cols]
            df3['delG'] = df2['delG']
#==============================================================================
#             if (method == 'GNB'):
#                 model = gausian_naive_bayes(df3,results)
#                 preds = model.predict(df3)
#             elif (method == 'DT'):
#                 model = decision_tree(df3,results)
#                 preds = model.predict(df3)
#             elif (method == 'SVM'):
#                 model = support_vector_machine(df3,results)
#                 preds = model.predict(df3)
#             elif (method == 'RF'):
#                 model = random_forest(df3,results)
#                 preds = model.predict(df3)
#==============================================================================
            if validation == 'NFCV':
                coor, mae = nfold_cv(df3, 10, method)
            elif validation == 'JKT':
                coor, mae = JK_Test(df3, method)
#            tp,tn,fp,fn = JK_Test(df3,method)
#            senstivity = float(tp)/float(tp+fn)
#            specificity = float(tn)/float(tn+fp)
#            accuracy2 = float(senstivity+specificity)/2
            if coor[0] - hold <= 0:
                df3 = df3.drop([cols], axis=1)
            else:
#                df5[cols] = df2[cols]
#                print cols, hold
                hold = coor[0]

        if hold == 0:
            hold = coor[0]
#    df5['delG'] = df2['delG']
    return df3
# =============================================================================
# def nfold_cv(df2,n,method='GNB'):
#     df3 = df2.sample(frac=1).reset_index(drop=True)
# #    print df3
#     length = len(df3)
#     test_len = length / n
#
#     x = 0
#     total_accuracy2 = 0
#     total_accuracy1 = 0
#     total_specificity = 0
#     total_senstivity = 0
#     for i in range(n):
#         df_test = df3[x:x+test_len]
#         hold1 = df3[0:x]
#         hold2 = df3[x+test_len:]
#         x = x+test_len
#         frames = [hold1, hold2]
#         df_train = pd.concat(frames)
#         train_results = df_train ['results']
#         test_results = df_test ['results']
# #        del df_train['results']
#         df_train = df_train.drop(['results'],axis = 1)
#         df_test = df_test.drop(['results'],axis = 1)
# #        del df_test['results']
#         if (method == 'GNB'):
#             model = gausian_naive_bayes(df_train,train_results)
#             pred1 = model.predict(df_test)

#         elif (method == 'DT'):
#             model = decision_tree(df_train,train_results)
#             pred1 = model.predict(df_test)
#         elif (method == 'SVM'):
#             model = support_vector_machine(df_train,train_results)
#             pred1 = model.predict(df_test)
#         elif (method == 'RF'):
#             model = random_forest(df_train,train_results)
#             pred1 = model.predict(df_test)
#         tp,tn,fp,fn = performance_estimate(pred1,test_results)
#         senstivity = float(tp)/float(tp+fn)
#         specificity = float(tn)/float(tn+fp)
#         accuracy = float(tp+tn)/float(tp+tn+fp+fn)
#         accuracy2 = float(senstivity+specificity)/2
#         total_accuracy1 += accuracy
#         total_accuracy2 += accuracy2
#         total_senstivity += senstivity
#         total_specificity += specificity
#     final_senstivity = total_senstivity / n
#     final_specificity = total_specificity / n
#     final_accuracy2 = total_accuracy2 / n
#     final_accuracy1 = total_accuracy1 / n
#     return final_senstivity, final_specificity, final_accuracy1, final_accuracy2
# =============================================================================


def feature_comb(df, feats, scale='log', method='LR'):
    results = df['delG']
    output = pd.DataFrame()
    temp = {}
    df_temp = df.drop(['delG'], axis=1)
#    print df_temp.columns
    f = itertools.combinations(df_temp, feats)
#    print f
    for cols in f:
        df2 = pd.DataFrame()
        temp = {}
        j = 1
        for i in cols:
            df2[i] = df[i]
            temp.update({'f'+str(j): i})
            j += 1

        df2['delG'] = results
        corr, mae = JK_Test(df2, method)
#        print df2
        df2_temp = df2.drop(['delG'], axis=1)
        
        if (method == 'LR'):
            model2 = linear_regression(df2_temp, results)
#            model_res =  model.fit()
#            print model.intercept_
            pred2 = model2.predict(df2_temp)



        elif (method == 'RR'):
            model2 = ridge_regression(df2_temp, results)
            pred2 = model2.predict(df2_temp)
        
        
#        model2 = linear_regression(df2_temp, results)
#        pred2 = model2.predict(df2_temp)
        sum_diff = 0
        for i in range(len(results)):
            if scale == 'log':
                sum_diff += abs(math.exp(abs(pred2[i])) - math.exp(abs(results[i])))
            else:
                sum_diff += abs(pred2[i] - results[i])
        mae2 = sum_diff / len(results)
        corr2 = pearsonr(pred2, results)

        temp.update({'corr': corr[0], 'mae': mae, 'p-value': corr[1],
                     'corr2': corr2[0], 'mae2': mae2, 'p-value2': corr2[1]})
#        print temp
        output = output.append(temp, ignore_index=True)
#    print output
    return output


def JK_Test(df, method='LR'):
    df2 = df.sample(frac=1).reset_index(drop=True)
    results = df2['delG']
#    print results
    df2 = df2.drop(['delG'], axis=1)
#    del df2['results']
    res2 = []
    sum_var = 0
    for i in range(len(df2)):
        df3 = pd.DataFrame()

        df3 = df3.append(df2.iloc[i])
        df4 = df2.drop(df2.index[i])

        result1 = results.iloc[i]
        result2 = results.drop(results.index[i])
#        print df3
        if (method == 'LR'):
            model = linear_regression(df4, result2)
#            model_res =  model.fit()
#            print model.intercept_
            pred1 = model.predict(df3)
            res2.append(pred1[0])

        elif (method == 'RR'):
            model = ridge_regression(df4, result2)
            pred1 = model.predict(df3)
            res2.append(pred1[0])

        sum_var = sum_var + abs(pred1[0] - result1)
#    train_res = model.predict(df2)
    corr = pearsonr(res2, results)

    mae = sum_var / len(df2)

    return corr, mae


def equation_calc(df, prop):
    y = pd.DataFrame()
    y_dict = {'res': 0}
    for x in df[prop]:
        y_dict['res'] = 13.8 - (0.91*math.pow(x, 0.5))
        y = y.append(y_dict, ignore_index=True)
    return y


def bootstraping(df, feats1, steps, scale='log', method='LR'):
    
    df2 = df.sample(frac=1).reset_index(drop=True)
#    print df2.columns
    df3 = pd.DataFrame()
    corr_boot = pd.DataFrame()
    corr2_boot = pd.DataFrame()
    mae_boot = pd.DataFrame()
    mae2_boot = pd.DataFrame()
    final_output1 = feature_comb(df2, feats1, scale, method)
    if steps > 1:
        for i in range(steps-1):
            df3 = pd.DataFrame()
            for j in range(len(df)):
                k = random.randint(0, len(df)-1)
                df3 = df3.append(df2.ix[k])
            df3 = df3.reset_index()
    #        print df3
            output = feature_comb(df2, feats1, scale, method)
    #        print output
            corr_boot['corr_'+str(i)] = output['corr']
            corr2_boot['corr2_'+str(i)] = output['corr2']
            mae_boot['mae_'+str(i)] = output['mae']
            mae2_boot['mae2_'+str(i)] = output['mae2']
            

        final_output1['corr'] = corr_boot.mean(axis=1)
        final_output1['corr2'] = corr2_boot.mean(axis=1)
        final_output1['mae'] = mae_boot.mean(axis=1)
        final_output1['mae2'] = mae2_boot.mean(axis=1)
#        elif (method == 'DT'):
#            model = decision_tree(df3,results)
#            preds = model.predict(df4)
#        elif (method == 'SVM'):
#            model = support_vector_machine(df3,results)
#            preds = model.predict(df4)
#        elif (method == 'RF'):
#            model = random_forest(df3,results)
#            preds = model.predict(df4)

#    for i in range(len(df2)):
#        print i
    return final_output1


'Pass complete dataframe in df1, already selected features in feat_set '
'without including delG and new feature to be added in new_feat'


def addonefeature(df1, feat_set, new_feat, scale='log', method='LR'):

    result_df = pd.DataFrame()

    for nf in new_feat:
        nf_set = list()
        nf_set = feat_set[:]
        nf_set.append(nf)
        nf_set.append('delG')
#        print nf_set
        df2 = df1[nf_set]
        out1 = feature_comb(df2, len(df2.columns)-1, scale, method)
#        print out1
        result_df = result_df.append(out1)
    return result_df
# data1 = pd.read_excel('Amit_14apr2017/All Features-Original3_norm.xlsx')


data1 = pd.read_csv('/home/ambuj/Documents/my/ProNIT/model_building/added_features_4jan19_removed_pairs_ln_delG.csv')
dg = data1['delG']
data1 = (data1 - data1.min())/(data1.max() - data1.min())
data1['delG'] = dg
data1 = data1.fillna(0)

# pred_len = equation_calc(data1, 'F20')
# data1['F25'] = pred_len['res']
num_steps = 1
nft = 1
bootstrap = 1
# csize = '_100-413'

validation = 'JKT'  # 'NFCV' for n-fold cross validation, 'JKT' for jack knife test



t1 = time.time()

#df3_out = frwd_sel_by_accuracy2(data1,method,validation,num_steps) #forward selection of variables
df3_out = data1


#corr_boot = pd.DataFrame()
#corr2_boot = pd.DataFrame()
#mae_boot = pd.DataFrame()
#mae2_boot = pd.DataFrame()
#final_output = feature_comb(data1,feats,method)
#for i in range(bootstrap-1):
#    output = feature_comb(data1,feats,method)
#    corr_boot['corr_'+str(i)] = output['corr']
#    corr2_boot['corr2_'+str(i)] = output['corr2']
#    mae_boot['mae_'+str(i)] = output['mae']
#    mae2_boot['mae2_'+str(i)] = output['mae2']
#if bootstrap > 1:
#    final_output['corr'] = corr_boot.mean(axis=1)
#    final_output['corr2'] = corr2_boot.mean(axis=1)
#    final_output['mae'] = mae_boot.mean(axis=1)
#    final_output['mae2'] = mae2_boot.mean(axis=1)

final_output = bootstraping(df3_out, nft, 1)
final_output2 = JK_Test(df3_out, method)
results2 = df3_out['delG']
#    print results
dfs2 = df3_out.drop(['delG'], axis=1)

m = linear_regression(dfs2, results2)
pred1 = m.predict(dfs2)
corr = pearsonr(pred1, results2)


t2 = time.time()
run_time = t2-t1
#final_output.to_csv('size_class/features_'+str(feats)+'_norm'+csize+'.csv')
#pred_len = equation_calc(data1, 'F20')

#print ("correlation = %f, \nMAE = %f  \nRuntime = %f"%
#       (corr[0], mae,run_time))
