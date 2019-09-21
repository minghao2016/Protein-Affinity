# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 13:08:34 2017

@author: Ambuj
"""
import sys
sys.path.insert(0,'/home/ambuj/Downloads/Project_work/simulation_22feb2018/')
import PDB2
import pandas as pd
import numpy as np
import glob

#def final_result (df_energy):
#    arranged = df_energy.sort_values(by =['prot_res','rna_res'])
#    sum_df = arranged.groupby(['prot_res','rna_res']).sum()
#    return sum_df

def df_append (df_energy, pdb_id, protein_chain, rna_chain, new_df):
    df_energy ['pdb_id'] = pdb_id
    df_energy ['prot_ch'] = protein_chain
    df_energy ['rna_ch'] = rna_chain
    
    new_df = new_df.append(df_energy)
    return new_df

def dot_inf (df_energy_all):
    dot_record = open('l9_p5_final_disorder_info_combined_nonribosomal.txt','r').readlines()
    df_energy_all ['DOT'] = False
    for i in range(0,len(dot_record),3):
        pdb_id = dot_record [i][1:5]
#        if pdb_id == '2B63':
        prot_ch = dot_record [i][6:7]
        for res_no in dot_record[i+1].split():
            res = res_no.split('_')[0]
            num = int(res_no.split('_')[1])
            dict_dot = {'pdb_id':pdb_id,'prot_ch':prot_ch,'res':res,'num':num}
#            print dict_dot,' '+str(dict_dot['num'])
            dl1 = df_energy_all.pdb_id == dict_dot['pdb_id'].lower()
            dl2 = df_energy_all.prot_ch == dict_dot['prot_ch'] 
            dl3 = df_energy_all.prot_ch == dict_dot['prot_ch'] 
            dl4 = df_energy_all.prot_res == dict_dot['res'] 
            dl5 = pd.to_numeric(df_energy_all.prot_resno, errors='coerce') == dict_dot['num']
            dl6 = dl1 & dl2 & dl3 & dl4 & dl5
            
            df_energy_all ['DOT'] = df_energy_all ['DOT'] | dl6
    return df_energy_all

def energy_matrix (df_energy_all):
    x1 = df_energy_all.groupby(['pdb_id','prot_ch', 'rna_ch','prot_res','rna_res','prot_resno','rna_resno']).sum()
#    x2 = x1
    x2 = x1.index
    x2 = x1.Vdw_energy
    x3 = x2.reset_index()
    x4 = x3.groupby(['prot_res','rna_res']).mean()
    x5 = x4.index
    x5 = x4.Vdw_energy
    energy_mat = np.zeros((20, 4))
    nt = {'A': 0, 'G':1, 'C':2, 'U':3}
    aa = {'ALA':0,'ARG':1,'ASN':2,'ASP':3, 'CYS':4, 'GLN':5, 'GLU':6, 'GLY':7, 'HIS':8, 'ILE':9, 'LEU':10, 'LYS':11, 'MET':12,'PHE':13, 'PRO':14, 'SER':15, 'THR':16, 'TRP':17, 'TYR':18, 'VAL':19}
    
    for aa_key,nt_key in x5.index:
        print x5[aa_key,nt_key]
        energy_mat [aa[aa_key]][nt[nt_key.strip()]] = x5[aa_key,nt_key]
    return energy_mat



#record = open('protein_rna_chain_inf_nonrb.txt','r').readlines()
#imp_lines = []
#for line in record:
#    if line[0:1] == '.':
#        imp_lines.append(line)

inst2 = PDB2.PDB()

intr_dict = {'OO':0}
inter_prot_rna = []
natype1 = 'DNA'

aa_20vdwrch = pd.read_csv('aa_20vdrch.csv')
if natype1 == 'DNA':
    param = pd.read_csv('dna_4vdrch.csv')
else:
    param = pd.read_csv('rna_4vdrch.csv')
#rna_4vdwrch = pd.read_csv('rna_4vdrch.csv')
#dna_4vdwrch = pd.read_csv('dna_4vdrch.csv')
final_energies = []
distance_cutoff = 8
df_energy_all = pd.DataFrame()
df_energy_break = pd.DataFrame()
for line in glob.glob('complex_str_with_chain/*.pdb'):
#    print (line)
    
    line_splt = line.split('/')
    na_chain = 'X'
    pdb_file = line_splt[1].rstrip()
    protein_chain = line_splt[1].split('_')[1][0:1]
    pdb_id = pdb_file[0:4]
    intdf_file = 'interaction_dataframe/'+pdb_file[0:-4]+'.dat'
    ans = pd.read_csv(intdf_file)
    if ans.empty:
        
        ans = inst2.proteinNAcontact('/home/ambuj/Documents/my/ProNIT/'+line,protein_chain,'X',distance_cutoff,natype='DNA')
        ans.to_csv(intdf_file)
        print 'empty data frame'
    else:
        
#            out_file = 'interaction_df/'+pdb_id+'_'+protein_chain+'_'+rna_chain+'_'+str(distance_cutoff)+'.csv'
#
#            ans.to_csv(out_file)
        

        int_dict1 = inst2.interaction_type(ans)
        inter_prot_rna.append([pdb_file, protein_chain, na_chain])
        
#        print int_dict1
        for key in int_dict1.keys():
            if key in intr_dict.keys():
                intr_dict[key] += int_dict1[key]
            else:
                intr_dict.update({key:int_dict1[key]})

        
        df_energy2 = inst2.energy(ans, aa_20vdwrch, param,natype='DNA')
        df_energy = df_energy2[df_energy2.Vdw_energy < 0]
        df_energy_break = df_energy_break.append(inst2.energy_div(df_energy), ignore_index = True)
        df_energy_all = df_append(df_energy, pdb_id, protein_chain, na_chain, df_energy_all)
        
#            sum_df = final_result (df_energy)
#            print df_energy_all
        final_energies.append(pdb_file+'\t'+protein_chain+'\t'+na_chain+'\t'+str(sum(df_energy.Vdw_energy)))
# =============================================================================
#     for rna_chain in line_splt[2].split():
#         pdb_file = line_splt[0][2:].rstrip()
# #        print pdb_file
#         protein_chain = line_splt[1]
# #        ans = inst2.proteinRNAcontact(pdb_file, protein_chain, rna_chain,distance_cutoff,False)
#         pdb_id = pdb_file.split('/')[1].split('.')[0]
#         in_file = 'interaction_df/'+pdb_id+'_'+protein_chain+'_'+rna_chain+'_'+str(distance_cutoff)+'.csv'
#         if os.path.exists(in_file):
#             
#             ans = pd.read_csv (in_file)
#         else:
#             ans = pd.DataFrame()
# #        bl =ans['distance'] > 2.5
# #        new_ans = ans[bl]
# #        n = ans[bl]
#         print pdb_file, protein_chain, rna_chain
#         if ans.empty:
#             print 'empty data frame'
#         else:
#             
# #            out_file = 'interaction_df/'+pdb_id+'_'+protein_chain+'_'+rna_chain+'_'+str(distance_cutoff)+'.csv'
# #
# #            ans.to_csv(out_file)
#             
# 
#             int_dict1 = inst2.interaction_type(ans)
#             inter_prot_rna.append([pdb_file, protein_chain, rna_chain])
#             
#             print int_dict1
#             for key in int_dict1.keys():
#                 if key in intr_dict.keys():
#                     intr_dict[key] += int_dict1[key]
#                 else:
#                     intr_dict.update({key:int_dict1[key]})
# 
#             
#             df_energy2 = inst2.energy(ans, aa_20vdwrch, rna_4vdwrch)
#             df_energy = df_energy2[df_energy2.Vdw_energy < 0]
#             
#             df_energy_all = df_append(df_energy, pdb_id, protein_chain, rna_chain, df_energy_all)
#             
# #            sum_df = final_result (df_energy)
# #            print df_energy_all
#             final_energies.append(pdb_file+'\t'+protein_chain+'\t'+rna_chain+'\t'+str(sum(df_energy.Vdw_energy)))
# =============================================================================
#        break
    
#    break
print final_energies
df_energy_all.to_csv('energy_matrix.csv')
#x = dot_inf(df_energy_all)
#x2 = x[x.DOT == True]
#x3 = x[x.DOT == False]
#energy_mat = energy_matrix(x)
#energy_mat2 = energy_matrix(x2)
#energy_mat3 = energy_matrix(x3)
#ptr = open('final_energies.txt','w')
#for line in final_energies:
#    ptr.write(line+'\n')
#ptr.close()