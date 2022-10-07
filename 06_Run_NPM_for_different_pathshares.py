import _DefaultValues
import pandas as pd
import time
import _NPM_egine_path_choice
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

class OptimizationModel(object):
    def __init__(self, para_npm, time_interval_demand,CM_avg_factor,used_att):
        self.para = para_npm
        date = self.para['date']
        self.DATE = date[1:len(date) - 1]
        sim_time_period = para_npm['sim_time_period'].split('-')
        time_start = int(pd.to_timedelta(sim_time_period[0]).total_seconds())
        time_end = int(pd.to_timedelta(sim_time_period[1]).total_seconds())
        self.opt_time_period = [time_start, time_end]
        self.PATH_ATTRIBUTE = pd.DataFrame()
        self.NODE = pd.DataFrame()
        self.DEMAND = pd.DataFrame()
        self.time_interval_demand = time_interval_demand
        self.input_file_path = 'Assignment_64800-68400_2017-03-16_syn/'
        self.biogeme_format = pd.DataFrame()
        self.CM_avg_factor = CM_avg_factor
        self.used_att = used_att
        self.Max_path_num = -1 #intialize
        # with open(self.input_file_path + 'P_constraints.pickle', 'rb') as handle:
        #     self.P_constraints = pickle.load(handle)


    def Load_input_files(self, TEST, Synthetic_data, path_file):
        # Variable settings
        if TEST == 1:
            self.opt_time_period = [int(pd.to_timedelta('7:00:00').total_seconds()),int(pd.to_timedelta('8:00:00').total_seconds())]
            input_file_path = 'Assignment_test/'

        else:
            input_file_path =  self.input_file_path
        self.PATH_ATTRIBUTE = pd.read_csv(input_file_path + 'tb_path_attribute.csv')
        self.NODE = pd.read_csv(input_file_path + 'tb_node_interval.csv')
        self.DEMAND = pd.read_csv(input_file_path + 'tb_demand_interval.csv')
        self.TEN_DEMAND = pd.read_csv(input_file_path + 'tb_demand.csv')
        if Synthetic_data == 1:
            self.TB_EXIT_DEMAND_raw = pd.read_csv(input_file_path + 'tb_exit_demand_synthesized2.csv')
        else:
            self.TB_EXIT_DEMAND_raw = pd.read_csv(input_file_path + 'tb_txn.csv')
            # process for the real-world test
            self.TB_EXIT_DEMAND_raw = self.TB_EXIT_DEMAND_raw.rename(columns = {'pax_tapin_time':'entry_time',\
            'pax_tapout_time':'exit_time', 'pax_origin':'origin', 'pax_destination':'destination'})
            self.TB_EXIT_DEMAND_raw['flow'] = 1
            self.TB_EXIT_DEMAND_raw = self.TB_EXIT_DEMAND_raw.drop(columns= ['user_id','tapout_ti'])


    def Load_input_files_SIM(self, TEST, Synthetic_data, path_file):

        # Variable settings
        path_external_data = 'External_data/'  # input data are in External_data folder
        if TEST == 1:
            self.opt_time_period = [int(pd.to_timedelta('7:00:00').total_seconds()),int(pd.to_timedelta('8:00:00').total_seconds())]
            input_file_path = 'Assignment_test/'
        else:
            input_file_path = self.input_file_path
        self.ITINERARY = pd.read_csv(input_file_path + 'tb_itinerary.csv')  # Itinerary table
        self.EVENTS = pd.read_csv(input_file_path + 'tb_event.csv')  # Event list
        self.CARRIERS = pd.read_csv(input_file_path + 'tb_carrier.csv', index_col=0)  # Carrier table
        self.QUEUES = pd.read_csv(input_file_path + 'tb_queue.csv', index_col=0)
        self.TB_TXN_RAW = pd.read_csv(input_file_path + 'tb_txn.csv')
        self.PATH_SHARE = pd.read_csv('Run_NPM_for_pathshares_res/' + path_file)
        self.PATH_SHARE.loc[self.PATH_SHARE['path_share']==0,'path_share'] = 0.01 # avoid boundary
        if Synthetic_data == 1:
            self.TRANSFER_WT = pd.read_csv(path_external_data + 'Transfer_Walking_Time.csv')
        else:
            self.TRANSFER_WT = pd.read_csv(path_external_data + 'Transfer_Walking_Time.csv')


        self.ACC_EGR_TIME = pd.read_csv(path_external_data + 'Access_egress_time.csv')
        self.NETWORK = pd.read_csv(input_file_path + 'tb_network.csv')  # Service network information
        self.OPERATION_ARRANGEMENT = pd.read_csv(path_external_data + 'Empty_Train_Arrangement.csv')
        # ****************process empty train*****************
        EMPTY_TRAIN_TIME_LIST = []
        operation_control = self.OPERATION_ARRANGEMENT[self.OPERATION_ARRANGEMENT.date == '[' + self.DATE + ']']
        if len(operation_control) > 0:
            operation_control.loc[:, 'dispatch_time'] = pd.to_timedelta(operation_control.time).dt.total_seconds()
            for index, operation_info in operation_control.iterrows():
                event_temp = self.EVENTS.copy()
                event_temp['carrier'] = event_temp['carrier_id'].apply(lambda x: x.split('_'))
                event_temp['line'] = event_temp['carrier'].apply(lambda x: int(x[0]))
                event_temp['dir'] = event_temp['carrier'].apply(lambda x: int(x[1]))
                event_temp = event_temp.loc[(event_temp['event_type'] == 1) &
                                            (event_temp['line'] == operation_info.line) &
                                            (event_temp['dir'] == operation_info.direction) &
                                            (event_temp['event_station'] == operation_info.station)]  # departure
                event_temp['time_diff'] = event_temp['event_time'] - operation_info.dispatch_time
                event_temp['time_diff'] = event_temp['time_diff'].abs()
                if len(event_temp) > 0:
                    min_index = event_temp['time_diff'].idxmin()
                    EMPTY_TRAIN_TIME_LIST.append(
                        (self.EVENTS.loc[min_index, 'carrier_id'], self.EVENTS.loc[min_index, 'event_time']))
        # ****************************************************

        self.EMPTY_TRAIN_TIME_LIST = EMPTY_TRAIN_TIME_LIST

        self.support_input = {'ITINERARY':self.ITINERARY, 'EVENTS':self.EVENTS, 'CARRIERS':self.CARRIERS, 'QUEUES':self.QUEUES,'TB_DEMAND':self.TEN_DEMAND ,
                              'PATH_ATTRIBUTE': self.PATH_ATTRIBUTE,'TRANSFER_WT':self.TRANSFER_WT,'ACC_EGR_TIME':self.ACC_EGR_TIME,'TB_TXN_RAW':self.TB_TXN_RAW ,
                             'NETWORK': self.NETWORK, 'OPERATION_ARRANGEMENT':self.OPERATION_ARRANGEMENT, 'EMPTY_TRAIN_TIME_LIST':self.EMPTY_TRAIN_TIME_LIST}

    def generate_Biogeme_files(self):
        path_share = self.PATH_SHARE
        path_set = path_share.loc[:, ['origin', 'destination', 'path_id']].drop_duplicates()
        path_set_more_than1 = path_set.groupby(['origin', 'destination']).size().reset_index(drop=False).rename(
            columns={0: 'num_of_path'})
        path_set_more_than1 = path_set_more_than1.loc[path_set_more_than1['num_of_path'] > 1]
        path_generate_old = path_share.merge(path_set_more_than1[['origin', 'destination']],
                                             left_on=['origin', 'destination'], right_on=['origin', 'destination'],
                                             how='inner')
        path_generate = pd.pivot_table(path_generate_old, index=['origin', 'destination', 'time_interval'],
                                       columns='path_id', values=['path_share'])
        path_generate.columns = ['path_' + str(col[1]) for col in path_generate.columns]
        path_num = len(path_generate.columns)
        self.Max_path_num = path_num
        path_list = list(path_generate.columns)
        path_generate = path_generate.reset_index(drop=False).fillna(0)
        melt_list = []
        for i in range(path_num):
            columns_name = 'passenger_' + str(i + 1)
            melt_list.append(columns_name)
            path_generate_old[columns_name] = i + 1
        passenger = pd.melt(path_generate_old,
                            id_vars=['origin', 'destination', 'path_id', 'time_interval', 'path_share'],
                            value_vars=melt_list)

        passenger = passenger.sort_values(by=['origin', 'destination', 'path_id', 'time_interval']).drop(
            columns=['variable']).rename(columns={'value': 'CHOICE'})

        passenger = passenger.loc[passenger['path_id'] == passenger['CHOICE']]

        for i in range(path_num):
            path_generate['AVAI_' + str(i + 1)] = (path_generate['path_' + str(i + 1)] > 0) * 1
        txn_choice = path_generate.merge(passenger[['origin', 'destination', 'time_interval', 'CHOICE','path_share']],
                                         left_on=['origin', 'destination', 'time_interval'],
                                         right_on=['origin', 'destination', 'time_interval']).drop_duplicates()
        txn_choice = txn_choice.drop(columns = ['path_' + str(i + 1) for i in range(path_num)])
        path_att_new = pd.pivot_table(self.PATH_ATTRIBUTE, index=['origin', 'destination', 'time_interval'], columns='path_id',
                                      values=self.used_att).reset_index(drop=False).fillna(0)

        path_att_new.columns = [col[0] + str(col[1]) for col in path_att_new.columns]
        input_data = txn_choice.merge(path_att_new, left_on=['origin', 'destination', 'time_interval'],
                         right_on=['origin', 'destination', 'time_interval'])
        self.biogeme_format = input_data.drop(columns = ['path_share'])


    def path_share_to_beta(self, path_share, demand):
        demand['time_interval'] = demand['entry_interval'] // _DefaultValues.TIME_INTERVAL_CHOICE * _DefaultValues.TIME_INTERVAL_CHOICE
        demand = demand.groupby(['origin','destination','time_interval']).sum()[['flow']].reset_index()
        data = self.biogeme_format.merge(path_share,left_on=['origin', 'destination','CHOICE' ,'time_interval'], right_on=\
            ['origin', 'destination','path_id' ,'time_interval']) # assign the path share
        data = data.merge(demand, left_on = ['origin','destination','time_interval'], right_on = ['origin','destination','time_interval'])
        data['WEIGHT'] = data['path_share'] * data['flow']
        data.columns = map(str.upper, data.columns)
        # from biogeme.expressions import *

        database = db.Database("est_beta", data)

        beta_dic = {}
        for name in self.used_att:
            beta_name = 'B_' + str.upper(name)
            beta_dic[beta_name] = bioexp.Beta(beta_name, 0, None,None, 0)

        V = {}
        av = {}
        for ind in range(1, self.Max_path_num+1):
            if ind not in V:
                V[ind] = 0
            for name in self.used_att:
                beta_name = 'B_' + str.upper(name)
                variable_name = str.upper(name) + str(ind)
                V[ind] += beta_dic[beta_name]*bioexp.Variable(variable_name)
            # define variables
            av_name = 'AVAI_'+str(ind)
            av[ind] = bioexp.Variable(av_name)
        CHOICE = bioexp.Variable('CHOICE')
        WEIGHT = bioexp.Variable('WEIGHT')


        logprob = bioexp.bioLogLogit(V, av, CHOICE)
        weight = WEIGHT
        formulas = {'loglike': logprob, 'weight': weight}
        biogeme = bio.BIOGEME(database, formulas)
        biogeme.modelName = "est_beta"
        results = biogeme.estimate()
        os.remove("est_beta.html")
        #os.remove("est_beta.pickle")
        # Print the estimated values
        betas = results.getBetaValues()
        beta={}
        for k, v in betas.items():
            beta[k] = v
        return beta

    def beta_to_path_share(self, beta, save = 0):


        data = self.PATH_ATTRIBUTE.copy()
        data['Utility_exp'] = 0
        for name in self.used_att:
            beta_name = 'B_' + str.upper(name)
            data['Utility_exp'] += data[name] * beta[beta_name]
        data['Utility_exp'] = np.exp(data['Utility_exp'])

        sum_uti = data.groupby(['origin', 'destination', 'time_interval']).sum().reset_index()[
            ['origin', 'destination', 'time_interval', 'Utility_exp']]. \
            rename(columns={'Utility_exp': 'sum_utility'})
        data = data.merge(sum_uti, left_on=['origin', 'destination', 'time_interval'],
                          right_on=['origin', 'destination', 'time_interval'])

        data['path_share'] = data['Utility_exp'] / data['sum_utility'] * 100
        data = data.loc[(data['time_interval']>=self.opt_time_period[0] - _DefaultValues.DEFAULT_WARM_UP_TIME*60)&
                        (data['time_interval'] < self.opt_time_period[1] + _DefaultValues.DEFAULT_COOL_DOWN_TIME * 60)]
        if save == 1:
            # output = self.PATH_SHARE.merge(data,left_on=['origin', 'destination', 'path_id', 'time_interval'],right_on=
            # ['origin', 'destination', 'path_id', 'time_interval'])
            # output['path_share'] = output['path_share_new']
            data.to_csv(self.input_file_path + 'tb_path_share_new.csv',index=False,columns=['origin', 'destination',\
                                                                                              'path_id', 'time_interval','path_share'])
        return data[['origin', 'destination', 'path_id', 'time_interval','path_share']]

    def generate_tb_exit_demand_synthesized(self, path_share, TEST, path_name):
        static = True
        # beta = [-0.0663, -0.438, -0.183,-0.941,-0.0767]
        beta = []
        OD_state, passenger_state, input_demand = _NPM_egine_path_choice.NPMModel(self.support_input, beta, path_share, static).run_assignment()
        OD_state = OD_state.loc[OD_state['flow'] > 0.1]
        passenger_state['arrival_time_interval'] = passenger_state['arrival_time'] // _DefaultValues.TIME_INTERVAL_DEMAND * _DefaultValues.TIME_INTERVAL_DEMAND
        passenger_state['departure_time_interval'] = passenger_state[
                                                       'departure_time'] // _DefaultValues.TIME_INTERVAL_DEMAND * _DefaultValues.TIME_INTERVAL_DEMAND
        passenger_state['Board_1st'] = 0
        passenger_state.loc[passenger_state['denied_boarding_times'] == 0, 'Board_1st'] = 1
        # arrival
        station_metrix = passenger_state.groupby(['departure_station', 'departure_line', 'departure_direction',
                                                  'arrival_time_interval']). \
            agg(
            {'Board_1st': 'sum', 'pax_number': 'sum'}).reset_index(drop=False)
        station_metrix = station_metrix.rename(
            columns={ 'pax_number': 'Arrivals', 'departure_station': 'StnID'})
        station_metrix['LB_rate'] = (station_metrix['Arrivals'] - station_metrix['Board_1st']) / station_metrix['Arrivals']
        station_metrix['LB_rate'] = station_metrix['LB_rate'].apply(lambda x: round(x, 4))
        # boarding
        station_metrix2 = passenger_state.groupby(['departure_station', 'departure_line', 'departure_direction',
                                                  'departure_time_interval']). \
            agg({'pax_number': 'sum'}).reset_index(drop=False)
        station_metrix2 = station_metrix2.rename(
            columns={ 'pax_number': 'Boardings', 'departure_station': 'StnID'})

        LB = {}
        Arr = {}
        Board = {}
        for info_time_interval in time_period_list:
            for info_s in info_station:
                LB[(info_time_interval,info_s)] = station_metrix.loc[(station_metrix['arrival_time_interval'] == info_time_interval) &
                                  (station_metrix['StnID'] == info_s[0]) &
                                  (station_metrix['departure_line'] == info_s[1]) &
                                  (station_metrix['departure_direction'] == info_s[2]), 'LB_rate'].values[0]

                Arr[(info_time_interval,info_s)] = station_metrix.loc[(station_metrix['arrival_time_interval'] == info_time_interval) &
                                           (station_metrix['StnID'] == info_s[0]) &
                                           (station_metrix['departure_line'] == info_s[1]) &
                                           (station_metrix['departure_direction'] == info_s[2]), 'Arrivals'].values[0]
                Board[(info_time_interval,info_s)] = station_metrix2.loc[(station_metrix2['departure_time_interval'] == info_time_interval) &
                                           (station_metrix2['StnID'] == info_s[0]) &
                                           (station_metrix2['departure_line'] == info_s[1]) &
                                           (station_metrix2['departure_direction'] == info_s[2]), 'Boardings'].values[0]


        out_put_file = 'Run_NPM_for_pathshares_res/'
        iter = 0
        RMSE = -1
        obj_value = -1
        df_dict = {'Iter': [iter+1], 'RMSE': [RMSE], 'Obj_value': [obj_value]}
        for info_time_interval in time_period_list:
            for info_s in info_station:
                name_LB = 'LB_' + str(info_s[0]) + '_' + str(info_s[1]) + '_' + str(info_s[2]) + '_' +  str(
                    info_time_interval)
                name_Arr = 'Arr_' + str(info_s[0]) + '_' + str(info_s[1]) + '_' + str(info_s[2]) + '_' +  str(
                    info_time_interval)
                name_Board = 'Board_' + str(info_s[0]) + '_' + str(info_s[1]) + '_' + str(info_s[2]) + '_' + str(
                    info_time_interval)
                df_dict[name_LB] = [LB[(info_time_interval, info_s)]]
                df_dict[name_Arr] = [Arr[(info_time_interval, info_s)]]
                df_dict[name_Board] = [Board[(info_time_interval, info_s)]]

        temp_records = pd.DataFrame(df_dict)
        path_name = path_name.split('_path_share_')[1]
        record_name = 'Model_records_'+path_name
        temp_records.to_csv(out_put_file + record_name, index=False)



        OD_state.to_csv(out_put_file + 'tb_exit_demand_'+ path_name, index=False)
        passenger_state.to_csv(out_put_file + 'tb_passenger_state'+ path_name, index=False)



# -----------------------------------------  Main Function ---------------------------------------------------

if __name__ == "__main__":

    #os.chdir('F:\[Network Performance Model]\Code') #change current file path
    para_list = pd.read_csv('0_user_configuration_parameter.csv').dropna()
    tic=time.time()
    for ix, para in para_list.iterrows():
        TEST = 0 # 0 = USE mtr network
        Synthetic_data = 0 # test with synthetic data
        time_interval_demand = 15*60 # resolution of exit OD
        sim_time_period = para['sim_time_period'].split('-')
        time_start = int(pd.to_timedelta(sim_time_period[0]).total_seconds())
        time_end = int(pd.to_timedelta(sim_time_period[1]).total_seconds())
        time_period_list = list(range(time_start, time_end, time_interval_demand))
        info_station = [(2,11,1)] #station + line + dir
        CM_avg_factor = 0
        path_att = []
        # ===============
        # 'Proposed': {'B_IN_VEHICLE_TIME': -0.156, 'B_NO_OF_TRANSFER': -0.544, 'B_TRANSFER_OVER_DIST': -1.291,
        #              'B_COMMONALITY_FACTOR': -3.413},
        # 'BYO': {'B_IN_VEHICLE_TIME': -0.205, 'B_NO_OF_TRANSFER': -1.218, 'B_TRANSFER_OVER_DIST': -2.499,
        #         'B_COMMONALITY_FACTOR': -6.184},
        # 'CORS': {'B_IN_VEHICLE_TIME': -0.231, 'B_NO_OF_TRANSFER': -1.189, 'B_TRANSFER_OVER_DIST': -2.316,
        #          'B_COMMONALITY_FACTOR': -6.537},
        scenario = {

            'Sun2012': [],
            'Zhao2017':[] }

        path_file_list = ['tb_path_share_' + ele + '.csv' for ele in scenario]
        for path_file in path_file_list:
            my_opt = OptimizationModel(para, time_interval_demand, CM_avg_factor, path_att)
            my_opt.Load_input_files(TEST = TEST, Synthetic_data = Synthetic_data,path_file = path_file)
            my_opt.Load_input_files_SIM(TEST = TEST, Synthetic_data = Synthetic_data,path_file = path_file)
            print('load file time:', time.time() - tic)
            #-----Generate new synthetic demand
            my_opt.generate_tb_exit_demand_synthesized(my_opt.PATH_SHARE, TEST = TEST, path_name = path_file)



    # tapin_csv_file.close()
    print('Total time:', time.time() - tic, 'sec')
    #x = input('Press any key to exit ...')