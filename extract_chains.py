#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 12:59:32 2018

@author: ambuj
"""
import pandas as pd
from Bio.PDB import *
io = PDBIO()
df1 = pd.read_csv('DATASET_23NOV2018.csv')
for i in range(len(df1)):
    pdbid = df1.loc[i,'PDB_COMPLEX'].split('_')[0]
    chain = df1.loc[i,'PDB_COMPLEX'].split('_')[1]
    structure = PDBParser(QUIET = True).get_structure(pdbid,'pdb_structures/'+pdbid.lower()+'.pdb')
    model = structure [0]
    ch1 = model[chain]
    io.set_structure(ch1)
    io.save(pdbid+'_'+chain+'.pdb')