from rdkit import Chem
from scopy.ScoDruglikeness.molproperty import *
from rdkit.Chem.rdMolDescriptors import *
from rdkit.Chem import Crippen, Descriptors

import pandas as pd
data = pd.read_csv('C:/Users/HyunJung Lee/OneDrive - 충남대학교/바탕 화면/CYP/HLM_data/physicochemistry.csv')
#print(data)
ID = data['ID']
NAME = data['Name']
SMILES = data['SMILES']
#print(SMILES)

Molecules = []
for smi in SMILES:
    m = Chem.MolFromSmiles(smi)
    Molecules.append(m)
    #print(Molecules)

Num_RotaB = []
HBA_values = []
HBD_values = []
Refractivity = []
PSA = []
logP = []

for m in Molecules:
    rota = CalcNumRotatableBonds(m)
    Num_RotaB.append(rota)
    hba = CalcNumHBA(m)
    HBA_values.append(hba)
    hbd = CalcNumHBD(m)
    HBD_values.append(hbd)
    refrac = Crippen.MolMR(m)
    Refractivity.append(refrac)
    psa = CalcTPSA(m)
    PSA.append(psa)
    mlogp = Crippen.MolLogP(m)
    logP.append(mlogp)

# print(Num_RotaB)
# print(HBA_values)
# print(HBD_values)
# print(Refractivity)
# print(PSA)
#print(logP)

from pandas import DataFrame
dataset={'ID':ID,'Name':NAME,'SMILES':SMILES,'Rotatable Bond count':Num_RotaB,'Hydrogen Acceptor count':HBA_values,
         'Hydrogen Donor count':HBD_values, 'Refractivity':Refractivity, 'Polar Surface Area':PSA, 'logP':logP}
data_df = DataFrame(dataset)
data_df.to_csv('C:/Users/HyunJung Lee/OneDrive - 충남대학교/바탕 화면/CYP/HLM_data/physicochemistry.csv')