import pandas as pd
import numpy as np


res = pd.read_csv('Model_records_2017-03-16_syn_1.csv')
K_b= 13
K_max = 15
col_beta =[]
for key in list(res.columns):
    if "B_" in key and "LB_" not in key:
        col_beta.append(key)

beta_res = res.loc[(res['Iter']>=K_b)&(res['Iter']<=K_max),col_beta].copy()

beta_dict = {}
for key in col_beta:
    beta_dict[key] = [np.mean(beta_res[key])]

beta_res_final = pd.DataFrame(beta_dict)

beta_res_final.to_csv('estimated_beta.csv',index=False)

a=1