#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 16:30:40 2018

@author: ambuj
"""
import pandas as pd
import sys
from sklearn import preprocessing
import math
sys.path.insert(0,'/home/ambuj/Documents/Prabakaran/')
import propylib_28Aug17
# =============================================================================
# 

def feature_extract(int_df1):
    all_feats = pd.read_csv('imp/extracted_feature_103.csv')
    three_one_aacode = {'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G','HIS':'H', 'ILE':'I', 'LYS':'K','LEU':'L','MET':'M','ASN':'N','PRO':'P', 'GLN':'Q','ARG':'R','SER':'S','THR':'T','VAL':'V','TRP':'W','TYR':'Y'}
    list_aa_pos = [x+'_'+str(int(y)) for x,y in set(zip(int_df1['prot_res'],int_df1['prot_resno']))]
    df_feats = pd.DataFrame()
    df_feats['aa_pos'] =list_aa_pos
    for f1 in range(103):
        col = 'F'+str(f1)
        df_feats[col] = ''
        for i in range (len(df_feats)):
            aa_pos1 = list_aa_pos[i]
            df_feats.ix[i,col] =all_feats.ix[f1 , three_one_aacode[aa_pos1.split('_')[0]]]
    return df_feats
def df_normalize (df1):
    x = df3.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    dfxx = pd.DataFrame(x_scaled)
    return dfxx
# =============================================================================    

#sys.path.insert(0,'/home/ambuj/Downloads/Project_work/simulation_22feb2018/')
#import PDB2

# =============================================================================
'DATASET TO INTERACTION DATAFRAME'
'OUTPUT KEPT IN int_df FOLDER'
#df1 = pd.read_csv('DATASET_23NOV2018.csv')
#prot_df = pd.DataFrame()
#dna_df = pd.DataFrame()
#dist = 8
#
#for i in range(len(df1)):
#    pdbid = df1.loc[i,'PDB_COMPLEX'].split('_')[0]
#    chain = df1.loc[i,'PDB_COMPLEX'].split('_')[1]
#    filename = 'pdb_structures/'+pdbid.lower()+'.pdb'
#    pdb_inst = propylib_28Aug17.PDB(filename)
#    
#    tp = pdb_inst.moleculetype()
#    prot_coords = pdb_inst.atom_coords(filename,chain)
#    for k in tp.keys():
#        if tp[k] == 'dna':
#            dna_coords = pdb_inst.atom_coords(filename,k)
#            result_int = pdb_inst.proteinNAcontact(filename, chain, k, dist)
#            result_int.to_csv('int_df/'+pdbid+'_'+chain+'_'+k+'_'+str(dist)+'.csv')
#             
# =============================================================================
'DATAFRAME TO PROPERTIES'

df1 = pd.read_csv('TEST_lt_50.csv')
dist =8
aa_20vdwrch = pd.read_csv('aa_20vdrch.csv')
rna_4vdwrch = pd.read_csv('rna_4vdrch.csv')
out_df = pd.DataFrame()
for i in range(len(df1)):
    pdbid = df1.loc[i,'PDB_COMPLEX'].split('_')[0]
    chain = df1.loc[i,'PDB_COMPLEX'].split('_')[1]
    filename = 'pdb_structures/'+pdbid.lower()+'.pdb'
    pdb_inst = propylib_28Aug17.PDB(filename)
    tp = pdb_inst.moleculetype()
    df_final = pd.DataFrame()
    count = 0
    for k in tp.keys():
#        print k
        if tp[k] == 'dna':
            print pdbid,chain,k
            
            int_df1 = pd.read_csv('int_df/'+pdbid+'_'+chain+'_'+k+'_'+str(dist)+'.csv')
            if len(int_df1) > 0:
                count += 1
                df2 = feature_extract(int_df1)
                df3 = df2.drop(['aa_pos'],1)
                df4 = df_normalize(df3)
                
                df4['protname'] =pdbid+'_'+chain+'_'+k
                df5 = df4.groupby(['protname']).mean()
                if len(df_final) == 0:
                    df_final = df5
                else:
                    for col2 in df_final.columns:
                        df_final.ix[0,col2] += df5.ix[0,col2]
    #            print df5
    df_final = df_final/count
    out_df = out_df.append(df_final)
out_df ['delG'] = list(df1['dG_avg'])
out_df.to_csv('TEST_lt_50_8A.csv')
#            dfxx = pdb_inst.energy(int_df1, aa_20vdwrch, rna_4vdwrch)
#    break
        
# =============================================================================
#     prot_inst = propylib_28Aug17.PDB(filename,chain)
#     prot_coords = prot_inst.atom.coords.data
#     print prot_coords
#     dist_prot_dna = []
#     for chain_type in tp.keys():
# #        print chain_type
#         
#         if tp[chain_type] == 'dna':
#             
#             dna_inst = propylib_28Aug17.PDB(filename,chain_type)
#             dna_coords = dna_inst.atom.coords.data
#             a = 0
#             for i in prot_coords:
# #                print float(i[0])
#                 b = 0
#                 for j in dna_coords:
#                     
#                     dist = math.sqrt( ((i[0]-j[0])*(i[0]-j[0])) +  ((i[1]-j[1])*(i[1]-j[1])) + ((i[2]-j[2])*(i[2]-j[2])))
#                     if dist < 10:
#                         print prot_inst.atom.resname[a], dna_inst.atom.resname[b]
#                     b += 1
#                 a += 1
# #            print cmap
# #    print pdbid,tp
# #    for dict1 in tp:
# #        print dict1
#     x = (pdb_inst.atom)
# =============================================================================
#    break
#    print (pdbid, chain)
