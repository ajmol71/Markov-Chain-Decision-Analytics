import numpy as np
import pandas as pd

countries = ["arg", "bol", "braz", "chi", "col", "ecu", "fguy", "guy", "para", "peru", "sur", "uru", "ven"]
ecu = ["peru", "col"]
peru = ["ecu", "bol", "braz", "chi", "col"]
col = ["ven", "ecu", "peru", "braz"]
ven = ["guy", "braz", "col"]
guy = ["ven", "sur", "braz"]
sur = ["fguy", "braz", "guy"]
fguy = ["braz", "sur"]
braz = ["fguy", "sur", "guy", "ven", "col", "peru", "bol", "para", "uru", "arg"]
bol = ["peru", "braz", "para", "chi", "arg"]
para = ["bol", "braz", "arg"]
chi = ["arg", "bol", "peru"]
arg = ["chi", "bol", "para", "uru", "braz"]
uru = ["arg", "braz"]
borders = [arg, bol, braz, chi, col, ecu, fguy, guy, para, peru, sur, uru, ven]

print("The {} Countries".format(str(len(countries))), countries)
print()
def pn(mat, n):
    mat = np.linalg.matrix_power(mat, n)
    return mat

prob = 1/len(countries)
InitialProbs = []
for i in range(0, len(countries)):
    InitialProbs.append(prob)

TransProbs = []
for i in range(0, len(borders)):
    row = []
    for j in countries:
        if j in borders[i]:
            row.append(1/len(borders[i]))
        else:
            row.append(0)
    TransProbs.append(row)

print("Transition Probability Matrix:\n", TransProbs)
print()

InitialMatrix = np.array(InitialProbs)
TransMatrix = np.array(TransProbs)

n = 100*365
SteadyState = pn(TransMatrix, n)
print("Long Term Probability (LTP) Matrix:")
print(SteadyState)

SteadyVec = np.matmul(InitialMatrix, SteadyState)

print()
for i in range(0, len(countries)):
    print("LTP of Country {}: {:3.2f}, or {:4.2f}%".format(countries[i], SteadyVec[i], 100*SteadyVec[i]))

MaxCountry = np.argmax(SteadyVec)

print("Most likely country: {}  with {:4.2f}% long-term likelihood".format(countries[MaxCountry], 100*SteadyVec[MaxCountry]))
