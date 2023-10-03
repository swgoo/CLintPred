import pandas
from rdkit import Chem
from scopy.ScoDruglikeness.molproperty import *
from rdkit.Chem.rdMolDescriptors import *
from rdkit.Chem import Crippen, Descriptors

import pubchempy as pcp

import pandas as pd
#data = pd.read_csv('C:/Users/HyunJung Lee/OneDrive - 충남대학교/바탕 화면/CYP/train2210.csv', encoding='euc-kr')
data = pd.read_excel('C:/Users/HyunJung Lee/OneDrive - 충남대학교/바탕 화면/data.xlsx',sheet_name=1)
print(data)
COMP = data['COMPOUND']
CAS = data['CAS']
CID = data['CID']
CHEMBL = data['CHEMBL']
logP = data['logP']
FUP = data['Fup']
CLinvit = data['Clint.invitro.']
CLinviv = data['Clint.invivo.']

isosmile = []
for i in CID:
    c=pcp.Compound.from_cid(i)
    iso_smi=c.isomeric_smiles
    isosmile.append(iso_smi)
#print(isosmile)


# cansmile = []
# isosmile = []
# for i in CID:
#     #m = Chem.MolFromSmiles(i)
#     can_smi = Chem.MolToSmiles(m,canonical=True)
#     iso_smi = Chem.MolToSmiles(m,isomericSmiles=True)
#     cansmile.append(can_smi)
#     isosmile.append(iso_smi)
# print(isosmile)

dataset={'Compound':COMP,'SMILES':isosmile, 'logP':logP, 'Fup':FUP, 'Clint.invitro.':CLinvit,'Clint.invivo.':CLinviv}
data_df = pandas.DataFrame(dataset)
data_df.to_csv('C://Users/HyunJung Lee/OneDrive - 충남대학교/바탕 화면/CID.csv')