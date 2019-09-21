#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 20:19:01 2018

@author: ambuj
"""
import pandas as pd
import numpy as np

def add_aa_prop(seq, prop):
    aa = {'A':0,'C':1,'D':2,'E':3, 'F':4, 'G':5, 'H':6, 'I':7, 'K':8, 'L':9, 'M':10, 'N':11, 'P':12,'Q':13, 'R':14, 'S':15, 'T':16, 'V':17, 'W':18, 'Y':19}
    aa_code = ['A','C','D','E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P','Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    comp_aa = {'A':0,'C':0,'D':0,'E':0, 'F':0, 'G':0, 'H':0, 'I':0, 'K':0, 'L':0, 'M':0, 'N':0, 'P':0,'Q':0, 'R':0, 'S':0, 'T':0, 'V':0, 'W':0, 'Y':0}
    prop_aa = {'1_propname':'','A':0,'C':0,'D':0,'E':0, 'F':0, 'G':0, 'H':0, 'I':0, 'K':0, 'L':0, 'M':0, 'N':0, 'P':0,'Q':0, 'R':0, 'S':0, 'T':0, 'V':0, 'W':0, 'Y':0}
    
    prop_df = pd.DataFrame()
    for k in seq.rstrip():
        if k in comp_aa.keys():
            comp_aa[k] += 1
    
    for j in range(len(prop)-1):
        
        prop_aa['1_propname'] = prop.iloc[j][0]
        propname = prop.iloc[j][0]
        total_value = 0
        for l in aa_code:
            prop_aa[l] = comp_aa[l]*prop.iloc[j][l]
            total_value += prop_aa[l]
        prop_dict1 = {'1_propname':propname, 'value':total_value}
        prop_df = prop_df.append(prop_dict1, ignore_index=True)
    return prop_df,sum(comp_aa.values())
        
            



record = pd.read_csv('dataset.csv')
details = pd.read_csv('detailed_dataset.csv')
record ['SEQUENCE'] = ''
for i in range(len(record)):
    df1 = details[record.iloc[i]['PDB_COMPLEX'] == details ['PDB_COMPLEX']]
    df2 = df1[record.iloc[i]['SEQUENCE_WILD1'] == df1 ['SEQUENCE_WILD1']]
    if len(df2) == 1:
        
        record.ix[i,'SEQUENCE'] = df2.iloc[0]['SEQUENCE']
        x=1
    elif len(df2) > 1:
        record.ix[i,'SEQUENCE'] = df2.iloc[0]['SEQUENCE']
        x=2
    else:
        print i
prop_table = pd.read_csv('imp/extracted_feature_103.csv')

df3 = record[record['SEQUENCE'] != '']
total_comp = {'A':0,'C':0,'D':0,'E':0, 'F':0, 'G':0, 'H':0, 'I':0, 'K':0, 'L':0, 'M':0, 'N':0, 'P':0,'Q':0, 'R':0, 'S':0, 'T':0, 'V':0, 'W':0, 'Y':0}
average_prop = pd.DataFrame(columns=prop_table.columns)
finaldataset = pd.DataFrame()
ctr = 0
for i in range(len(df3)):
    prop_df, length_seq = add_aa_prop(df3.iloc[i]['SEQUENCE'], prop_table)
    prop_df = prop_df.set_index('1_propname')
    average_prop = dict.fromkeys(prop_df.index)
    for n in average_prop.keys():
#        print prop_df.ix[n]['value']
        average_prop[n] =  float(prop_df.ix[n]['value'])/ float(length_seq)
    average_prop.update({'delG':float(df3.iloc[i]['dG_avg'])})
    finaldataset = finaldataset.append(average_prop,ignore_index=True)
    finaldataset.to_csv('Final_dataset_98complexes.csv')
#    break
#    else:
#        for o in range(len(prop_df)):
#            for p in composition.keys():
#                total_prop.ix[o][p] += prop_df.iloc[o][p] 