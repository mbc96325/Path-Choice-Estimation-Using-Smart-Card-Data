import os
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import math
from datetime import datetime, timedelta
from sklearn.preprocessing import normalize
colors = ["#3366cc", "#dc3912", "#ff9900", "#109618", "#990099"]

def process(data,max_iter):
    x =[]
    y =[]
    count = 0
    for idx, info in data.iterrows():
        count +=1
        if count > max_iter:
            break
        x.append(count)
        if len(y) == 0:
            y.append(info['target'])
        else:
            if info['target']< y[-1]:
                y.append(info['target'])
            else:
                y.append(y[-1])
    return np.array(x),np.array(y)
def obj_func_curve(model_record, save_fig,max_iter):


    fig, ax1 = plt.subplots(figsize=(10, 5))
    font_size = 17
    ax1.set_ylabel('Objective Function ($10^3$)',fontsize=font_size)
    ax1.set_xlabel('Number of Function Evaluations',fontsize=font_size)
    key_list=['NMSA','MADS','SPSA','BYO','CORS']
    color_id = 0
    for key in key_list:
        x,y = process(model_record[key],max_iter)
        ax1.plot(x,y/1000,marker='*',markersize=3,color=colors[color_id],linewidth=1.0,label=key)
        color_id+=1


    ax1.tick_params(axis='y', labelsize = font_size)
    ax1.tick_params(axis='x', labelsize=font_size)
    ax1.legend(fontsize = 15)
    # ax1.set_ylim(left, right)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    if save_fig==0:
        plt.show()
    else:
        plt.savefig('Obj_func.png', dpi=200)


if __name__ == '__main__':
    Bay = pd.read_csv('Results/Bayesian_opt_process_EI.csv')
    Bay['target'] = -Bay['target']
    model_records = {'BYO':Bay}
    CORS = pd.read_csv('Results/CORS_RBF_results.csv')
    #CORS = CORS.rename(columns = {'f_value':'target'})
    model_records['CORS']=CORS
    NMSA = pd.read_csv('Results/NelderMead_results.csv')
    SPSA = pd.read_csv('Results/SPSA_results2.csv')
    MADS = pd.read_csv('Results/Nomad_output.csv')
    model_records['NMSA'] = NMSA
    model_records['SPSA'] = SPSA
    model_records['MADS'] = MADS
    obj_func_curve(model_records, save_fig = 1,max_iter = 100)

