import pandas as pd
import numpy as np
import pickle
import _DefaultValues
import time

def beta_to_path_share(PATH_ATTRIBUTE, beta, used_att):

    data = PATH_ATTRIBUTE.copy()
    data['Utility_exp'] = 0
    ind = 0
    for name in used_att:
        # beta_name = 'B_' + str.upper(name)
        data['Utility_exp'] += data[name] * beta[ind]
        ind += 1
    data['Utility_exp'] = np.exp(data['Utility_exp'])


    sum_uti = data.groupby(['origin', 'destination']).sum().reset_index()[
        ['origin', 'destination', 'Utility_exp']]. \
        rename(columns={'Utility_exp': 'sum_utility'})
    data = data.merge(sum_uti, left_on=['origin', 'destination'],
                      right_on=['origin', 'destination'])

    data['path_share'] = data['Utility_exp'] / data['sum_utility'] * 100
    return data

def sample_beta_group_OD(path_att, beta_bound, used_att, input_file_path):
    N = 10 # number of replications, cannot be too large otherwise memory error
    beta = np.zeros(len(beta_bound))
    path_share = pd.DataFrame()
    for i in range(N):
        print (i)
        for j in range(len(beta_bound)):
            beta[j] = np.random.uniform(beta_bound[j][0], beta_bound[j][1], 1)
        path_share_temp = beta_to_path_share(path_att, beta, used_att)
        path_share_temp['path_share'] = path_share_temp['path_share'].apply(lambda x: round(x, 1)) # 3 digital percision
        path_share_temp = path_share_temp.rename(columns = {'path_share':'path_share' + str(i)})
        if i == 0:
            path_share = path_share_temp.copy()
            path_share['index'] = path_share['path_share' + str(i)].astype('str')
        else:
            path_share = path_share.merge(path_share_temp[['origin', 'destination','path_id','path_share' + str(i)]],
                             left_on = ['origin', 'destination','path_id'],
                             right_on = ['origin', 'destination','path_id'])
            path_share['index'] = path_share['index'] + '_' + path_share['path_share' + str(i)].astype('str')
    # path_share.to_csv('constraints.csv')
    path_share_group = path_share.groupby('index')
    P_constraints = {}
    group_id = 0
    for key, info in path_share_group:
        if len(info)>=2: # add constraints
            group_id+=1
            equal_odr_list = info[['origin','destination','path_id']].values.tolist()
            P_constraints[group_id] = equal_odr_list
    print ('total group:', group_id)
    with open(input_file_path + 'P_constraints.pickle', 'wb') as handle:
        pickle.dump(P_constraints, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #-------


def sample_beta_group_OD_more(path_att, beta_bound, used_att, input_file_path):
    N = 5 # number of replications, cannot be too large otherwise memory error
    beta = np.zeros(len(beta_bound))
    path_share = pd.DataFrame()
    for i in range(N):
        print (i)
        for j in range(len(beta_bound)):
            beta[j] = np.random.uniform(beta_bound[j][0], beta_bound[j][1], 1)
        path_share_temp = beta_to_path_share(path_att, beta, used_att)
        path_share_temp['path_share'] = path_share_temp['path_share'].apply(lambda x: round(x, 1)) # 3 digital percision
        path_share_temp = path_share_temp.rename(columns = {'path_share':'path_share' + str(i)})
        if i == 0:
            path_share = path_share_temp.copy()
            path_share['index'] = path_share['path_share' + str(i)].astype('str')
        else:
            path_share = path_share.merge(path_share_temp[['origin', 'destination','path_id','path_share' + str(i)]],
                             left_on = ['origin', 'destination','path_id'],
                             right_on = ['origin', 'destination','path_id'])
            path_share['index'] = path_share['index'] + '_' + path_share['path_share' + str(i)].astype('str')
    # path_share.to_csv('constraints.csv')
    path_share_group = path_share.groupby('index')
    P_constraints = {}
    group_id = 0
    for key, info in path_share_group:
        if len(info)>=2: # add constraints
            group_id+=1
            equal_odr_list = info[['origin','destination','path_id']].values.tolist()
            P_constraints[group_id] = equal_odr_list
    print ('total group:', group_id)
    with open(input_file_path + 'P_constraints_more.pickle', 'wb') as handle:
        pickle.dump(P_constraints, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #-------


if __name__ == '__main__':

    para_list = pd.read_csv('0_user_configuration_parameter.csv').dropna()
    tic=time.time()
    TEST = 0
    date = '2017-03-16'
    name = '_syn'
    for ix, para in para_list.iterrows():
        sim_time_period = para['sim_time_period'].split('-')
        time_start = int(pd.to_timedelta(sim_time_period[0]).total_seconds())
        time_end = int(pd.to_timedelta(sim_time_period[1]).total_seconds())
        input_file_path = 'Assignment_' + str(time_start) + '-' + str(time_end) + '_' + date + name +  '/'
        path_att = pd.read_csv(input_file_path + 'tb_path_attribute.csv')
        path_att = path_att.drop(columns = ['time_interval']) # not consider time interval at this time
        # used_att = ['in_vehicle_time','no_of_transfer','transfer_time','commonality_factor']
        used_att = _DefaultValues.path_att
        path_att = path_att.loc[:,['origin','destination','path_id']+used_att].drop_duplicates()
        beta_bound = [[-0.2, 0], [-1, 0], [-3, 0],[-5,0]]
        # sample_beta_group_OD(path_att, beta_bound, used_att, input_file_path)
        sample_beta_group_OD_more(path_att, beta_bound, used_att, input_file_path)
    print('Add constraints time:', time.time() - tic, 'sec')