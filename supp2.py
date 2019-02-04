#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 01:30:33 2018

@author: marcello
"""

from utils import *
import pydicom

def parseDir(dr):
    RCC = dirNoDots(dr)
    lis = []
    nmLis = []
    RCC = [elt for elt in RCC if ('.' not in elt and '\r' not in elt)]
    RCC = [elt for elt in RCC if hasNumbers(elt)]
    RCC = [elt for elt in RCC if 'idney' in elt or 'ideny' in elt]
    RCC = [elt for elt in RCC if '_' not in elt]
    
    dicsLis = []
    
    folLis = []
    imsLis,refLis = [],[]
    for f in RCC:
        imsF = dr + '/' + f
        print(imsF)
        ims = dirNoDots(imsF)
        ims = [elt for elt in ims if '.' not in elt]
        if ims:
            folLis.append(f)
            seqList = []
            dics = []
            patient_refs = []
            for im in ims:
                #print(im)
                nrrdFol = imsF + '/' + im + '/'
                nrrdFls = dirNoDots(nrrdFol)
                nrrds = [fl for fl in nrrdFls if fl.endswith('.seg.nrrd')]
                nrrds_all = [fl for fl in nrrdFls if fl.endswith('.nrrd')]
                not_nrrds = [fl for fl in nrrdFls if 'nrrd' not in fl]
                refs = pydicom.read_file(nrrdFol+not_nrrds[0])
                patient_refs.append(refs)
                dics.append(im)
            imsLis.append(seqList)
            dicsLis.append(dics)
            refLis.append(patient_refs)
    return dicsLis, lis, imsLis, folLis, refLis


nm, demo = readCSVs('../clinical_data_renal_tumor.csv')
nm = list(nm)

drs = ['Penn_renal_tumor_segment_relabeled',
'TCGA',
'Xiangya Second Hospital',
'Hunan People Hospital']
drs = ['../../' + elt for elt in drs]
imsLis, folLis, dicLis,refLis  = [], [], [],[]
for dr in drs:
    dics, _, dirIms, dirFols,refers = parseDir(dr)
    imsLis = imsLis + dirIms
    folLis = folLis + dirFols
    dicLis = dicLis + dics
    refLis = refLis + refers
    
with open('Ims_binned') as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
#content = [x.strip() for x in content if 'TYPE' not in x and len(x)>=1]
content = [x.strip() for x in content]
inds = [content.index(x) for x in content if 'TYPE' in x]  
'''


trn = content[1:inds[1]-2]
val = content[inds[1]+1:inds[2]-1]
tst = content[inds[2]+1:]




out = [[],[],[]]
for i,ls in enumerate([trn,val,tst]):
    for elt in ls:
        
        if 'penn' in elt.lower():
            fl=[ff for ff in folLis2 if (('penn' in ff) and (elt[-3:] in ff))][0]
            ind = folLis2.index(fl)
        else:
            
            
            ind = folLis2.index(elt.lower())
        r = refLis[ind]
        sl=r.SliceThickness
        rp=r.RepetitionTime
        ec=r.EchoTime
        try:
            ms=r.MagneticFieldStrength
            tens=10**np.floor(np.log10(ms))
            ms = ms/tens
        except Exception as e:
            ms=-1

        
        out[i].append([sl,rp,ec,ms])
'''
        
##version 2!!!
'''
______________________________
'''
folLis2 = [f.lower() for f in folLis]
tmp=list(map(lambda x: [elt for elt in folLis2 if x in elt],['penn','xy','hp','tcga']))
out = [[],[],[],[]]
for i,ls in enumerate(tmp):
    for elt in ls:
        
        if 'penn' in elt.lower():
            fl=[ff for ff in folLis2 if (('penn' in ff) and (elt[-3:] in ff))][0]
            ind = folLis2.index(fl)
        else:
            
            
            ind = folLis2.index(elt.lower())
        ref = refLis[ind]
        im = dicLis[ind]
        
        tpL=[]
        
        for tp in ['T1C','T2']:
            tp_ind = [i for i,t in enumerate(im) if tp.lower() in t.lower().lower()][0]
            r = ref[tp_ind]
            sl=r.SliceThickness
            rp=r.RepetitionTime
            ec=r.EchoTime
            try:
                ms=r.MagneticFieldStrength
                tens=10**np.floor(np.log10(ms))
                ms = ms/tens
            except Exception as e:
                ms=-1
            '''
            if ms<10 and ms!=-1:
                Decimal('40800000000.00000000000000')
                
            '''
            tpL.append([sl,rp,ec,ms])
            '''
        except Exception as e:
            print(elt,tp)
            '''
            
        out[i].append(tpL)
tmps=[]
for o in out:
    ar = np.vstack(o)
    ls=[]
    for row in ar.transpose():
        nrow = row[row!=-1]
        ls.append([np.mean(nrow),
         np.std(nrow),
         np.max(nrow),
         np.min(nrow),
         np.percentile(nrow, 25),
         np.percentile(nrow, 50),
         np.percentile(nrow, 75)])
    tmp = np.vstack(ls)
    tmps.append(tmp)
    
    
import csv

with open('raw_inst_data.csv', mode='w') as employee_file:
    employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for o in out:
        for i in range(2):
            
            #print(i)
            ar = np.vstack([e[i] for e in o])
            for row in ar.transpose():
                employee_writer.writerow(row)

with open('employee_file.csv', mode='w') as employee_file:
    employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for tmp in tmps:
        for row in tmp:
            
            employee_writer.writerow(row)