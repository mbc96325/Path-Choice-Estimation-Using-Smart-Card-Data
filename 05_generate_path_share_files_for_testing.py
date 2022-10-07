import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import _DefaultValues
from _postprocess_mtr_network_operation_ver2 import post_process

def ChangeCEN_ADM_fraction(data,transfer_CEN):
    route_raw = pd.read_csv('External_data/mtr_network_operation_assignment_InclCEN40.csv')
    route_raw = route_raw.loc[:,['ORI_STN_NO', 'DES_STN_NO', 'HARBOUR_CO', 'processed']].drop_duplicates()
    len1 = len(data)
    data = data.merge(route_raw[['ORI_STN_NO','DES_STN_NO','HARBOUR_CO','processed']],left_on = ['origin','destination','path_id'],
                      right_on = ['ORI_STN_NO','DES_STN_NO','HARBOUR_CO'])
    if len(data)<len1:
        print('lose paths, please check')
    #processed=0, paths not transfer at CEN and ADM
    #processed=1, paths not transfer at ADM
    #processed=2, paths not transfer at CEN new added
    data.loc[data['processed'] == 1,'path_share'] = (1-transfer_CEN)*100
    data.loc[data['processed'] == 2, 'path_share'] = transfer_CEN * 100
    path_set = data.loc[:,['origin','destination','time_interval','path_id','path_share']].drop_duplicates()
    path_set['sum_path_share'] = path_set.groupby(['origin','destination','time_interval'])['path_share'].transform(sum)
    path_set['factor_correct'] = 100/path_set['sum_path_share']
    path_set['path_share'] *= path_set['factor_correct']
    #check
    path_set['sum_path_share'] = path_set.groupby(['origin', 'destination', 'time_interval'])['path_share'].transform(
        sum)
    path_set_error = path_set.loc[path_set['sum_path_share']>100.1]
    if len(path_set_error)>0:
        print('error in path share correct, please check')
        print(path_set_error)
    data = data.drop(columns = ['path_share','ORI_STN_NO','DES_STN_NO','HARBOUR_CO','processed'])
    len1 = len(data)
    data = data.merge(path_set[['origin','destination','path_id','time_interval','path_share']],left_on = ['origin','destination','path_id','time_interval'],
                      right_on = ['origin','destination','path_id','time_interval'])
    if len(data)<len1:
        print('lose paths, please check 2')
    #a=11
    return data

def beta_to_path_share(PATH_ATTRIBUTE, beta, input_file_path,used_att, out_put_period, network_fil, path_file_name):
    data = PATH_ATTRIBUTE.copy()
    data['Utility_exp'] = 0
    for name in used_att:
        beta_name = 'B_' + str.upper(name)
        data['Utility_exp'] += data[name] * beta[beta_name]
    data['Utility_exp'] = np.exp(data['Utility_exp'])

    sum_uti = data.groupby(['origin', 'destination', 'time_interval']).sum().reset_index()[
        ['origin', 'destination', 'time_interval', 'Utility_exp']]. \
        rename(columns={'Utility_exp': 'sum_utility'})
    data = data.merge(sum_uti, left_on=['origin', 'destination', 'time_interval'],
                      right_on=['origin', 'destination', 'time_interval'])


    data['path_share'] = data['Utility_exp'] / data['sum_utility'] * 100
    data = data.loc[:,['origin', 'destination','path_id','path_share']].drop_duplicates()
    data['key'] = 0
    time_period_list = list(range(out_put_period[0],out_put_period[1], _DefaultValues.CHOICE_INTERVAL))
    time_list = pd.DataFrame({'key':[0]*len(time_period_list), 'time_interval':time_period_list})
    data = data.merge(time_list,left_on = ['key'],right_on = ['key']).drop(columns = ['key'])
    return data


if __name__ == '__main__':
    para_list = pd.read_csv('0_user_configuration_parameter.csv').dropna()
    for ix, para in para_list.iterrows():
        TEST = 0 # 0 = USE mtr network
        Synthetic_data = 1 # test with synthetic data
        time_interval_demand = 15*60 # resolution of exit OD
        sim_time_period = para['sim_time_period'].split('-')
        time_start = int(pd.to_timedelta(sim_time_period[0]).total_seconds())
        time_end = int(pd.to_timedelta(sim_time_period[1]).total_seconds())
        time_period_list = list(range(time_start, time_end, time_interval_demand))
        info_station = [(2,11,1)] #station + line + dir
        opt_time_period = [time_start, time_end]
        input_file_path = 'Assignment_' + str(opt_time_period[0]) + '-' + str(
            opt_time_period[1]) + '_Car_Cap_' + str(_DefaultValues.DEFAULT_CAR_CAP_1) + '/'
    path_att = pd.read_csv('Assignment_64800-68400_2017-03-16_syn/tb_path_attribute.csv')
    # user_configuration = '0_user_configuration_parameter.csv'
    # mtr_raw_assignment_file_path = 'External_data/mtr_network_operation_assignment.csv'
    # network_file = post_process(user_configuration, mtr_raw_assignment_file_path)
    network_file = []
    # used_att = ['in_vehicle_time', 'no_of_transfer', 'transfer_time',
    #             'commonality_factor']
    used_att = _DefaultValues.path_att # names need to consistent with path at


    zhao_est = pd.read_csv('Probabilistic_benchmark_models/output/estimated_paras_Geo.csv')
    zhao_dict = {'B_IN_VEHICLE_TIME':zhao_est['Est'].iloc[0], 'B_NO_OF_TRANSFER':zhao_est['Est'].iloc[1],'B_TRANSFER_OVER_DIST':zhao_est['Est'].iloc[2], 'B_COMMONALITY_FACTOR':zhao_est['Est'].iloc[3]}

    sun_est = pd.read_csv('Probabilistic_benchmark_models/output/estimated_paras.csv')
    sun_dict = {'B_IN_VEHICLE_TIME':sun_est['Est'].iloc[0], 'B_NO_OF_TRANSFER':sun_est['Est'].iloc[1],'B_TRANSFER_OVER_DIST':sun_est['Est'].iloc[2], 'B_COMMONALITY_FACTOR':sun_est['Est'].iloc[3]}
    scenario = {'Proposed': {'B_IN_VEHICLE_TIME':-0.156, 'B_NO_OF_TRANSFER':-0.544,'B_TRANSFER_OVER_DIST':-1.291, 'B_COMMONALITY_FACTOR':-3.413},
                'BYO':{'B_IN_VEHICLE_TIME':-0.205, 'B_NO_OF_TRANSFER':-1.218,'B_TRANSFER_OVER_DIST':-2.499, 'B_COMMONALITY_FACTOR':-6.184},
                'CORS':{'B_IN_VEHICLE_TIME':-0.231, 'B_NO_OF_TRANSFER':-1.189,'B_TRANSFER_OVER_DIST':-2.316, 'B_COMMONALITY_FACTOR':-6.537},
                'Sun2012':sun_dict,
                'Zhao2017':zhao_dict}


    #set_up_output_period

    time_peiod = ['18:00:00','19:00:00']
    time_start = int(pd.to_timedelta(time_peiod[0]).total_seconds())
    time_end = int(pd.to_timedelta(time_peiod[1]).total_seconds())    
    
    warm_cool = [60,60]
    out_put_period = [time_start-warm_cool[0]*60, time_end+warm_cool[1]*60]


    for path_file_name in scenario:
        # path_file_name = 'Estimated'
        beta = scenario[path_file_name]
        data = beta_to_path_share(path_att, beta, input_file_path, used_att, out_put_period, network_file, path_file_name)

        # transfer_CEN = 0.4
        # data = ChangeCEN_ADM_fraction(data,transfer_CEN=transfer_CEN)

        data.to_csv('Run_NPM_for_pathshares_res/' + 'tb_path_share_' + path_file_name + '.csv', index=False,
                    columns=['origin', 'destination',
                             'path_id', 'time_interval',
                             'path_share'])
        print('generate finish, see tb_path_share_'+path_file_name)

