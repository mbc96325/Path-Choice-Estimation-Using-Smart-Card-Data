import _DefaultValues
import pandas as pd
import time
import biogeme.database as db
import biogeme.biogeme as bio
import biogeme.expressions as bioexp
from biogeme import models
import _Subproblem1
import _NPM_egine_path_choice
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

class OptimizationModel(object):
    def __init__(self, para_npm, time_interval_demand,CM_avg_factor,used_att,TEST, file_tail_name, date, name):
        self.para = para_npm
        self.DATE = date
        sim_time_period = para_npm['sim_time_period'].split('-')
        time_start = int(pd.to_timedelta(sim_time_period[0]).total_seconds())
        time_end = int(pd.to_timedelta(sim_time_period[1]).total_seconds())
        self.opt_time_period = [time_start, time_end]
        self.PATH_ATTRIBUTE = pd.DataFrame()
        self.NODE = pd.DataFrame()
        self.DEMAND = pd.DataFrame()
        self.time_interval_demand = time_interval_demand
        self.input_file_path = 'Assignment_' + str(self.opt_time_period[0]) + '-' + str(
            self.opt_time_period[1]) + '_' +date + name + '/'
        self.biogeme_format = pd.DataFrame()
        self.CM_avg_factor = CM_avg_factor
        self.used_att = used_att
        self.Max_path_num = -1 #intialize
        self.tail_name = file_tail_name
        if TEST == 1:
            self.P_constraints = {}
        else:
            with open(self.input_file_path + 'P_constraints.pickle', 'rb') as handle:
                self.P_constraints = pickle.load(handle)


    def Load_input_files(self, TEST, Synthetic_data, generate_data):
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
        if not generate_data:
            if Synthetic_data == 1:
                self.TB_EXIT_DEMAND_raw = pd.read_csv(input_file_path + 'tb_exit_demand_synthesized2.csv')
            else:
                self.TB_EXIT_DEMAND_raw = pd.read_csv(input_file_path + 'tb_txn.csv')
                # process for the real-world test
                self.TB_EXIT_DEMAND_raw = self.TB_EXIT_DEMAND_raw.rename(columns = {'pax_tapin_time':'entry_time',\
                'pax_tapout_time':'exit_time', 'pax_origin':'origin', 'pax_destination':'destination'})
                self.TB_EXIT_DEMAND_raw['flow'] = 1
                self.TB_EXIT_DEMAND_raw = self.TB_EXIT_DEMAND_raw.drop(columns= ['user_id','tapout_ti'])


    def Load_input_files_SIM(self, TEST, Synthetic_data, generate_data):

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

        if TEST == 1:
            self.TRANSFER_WT = pd.read_csv(path_external_data + 'Transfer_Walking_Time_TEST.csv')
            self.OPERATION_ARRANGEMENT = pd.read_csv(path_external_data + 'Empty_Train_Arrangement_TEST.csv')
            self.PATH_SHARE = pd.read_csv(input_file_path + 'tb_path_share.csv')
            self.PATH_REFER = pd.read_csv(path_external_data + 'tb_path_share_refer_TEST.csv')
        else:
            self.TRANSFER_WT = pd.read_csv(path_external_data + 'Transfer_Walking_Time.csv')
            self.OPERATION_ARRANGEMENT = pd.read_csv(path_external_data + 'Empty_Train_Arrangement.csv')
            if Synthetic_data == 1:
                if not generate_data:
                    self.PATH_SHARE = pd.read_csv(input_file_path + 'tb_path_share_synthetic.csv')
                    self.PATH_REFER = self.PATH_SHARE.copy()
            else:
                self.PATH_SHARE = pd.read_csv(input_file_path + 'tb_path_share.csv')
                self.PATH_REFER = pd.read_csv(path_external_data + 'tb_path_share_refer.csv')
        if not generate_data:
            self.PATH_REFER = self.PATH_REFER.loc[self.PATH_REFER['time_interval'] == self.opt_time_period[0]]
            self.PATH_REFER = self.PATH_REFER.drop(columns=['time_interval'])
            # self.TRANSFER_WT = self.TRANSFER_WT.iloc[0:0,:]
            self.PATH_SHARE.loc[self.PATH_SHARE['path_share'] == 0, 'path_share'] = 0.01  # avoid boundary

        self.ACC_EGR_TIME = pd.read_csv(path_external_data + 'Access_egress_time.csv')
        self.NETWORK = pd.read_csv(input_file_path + 'tb_network.csv')  # Service network information

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
        data = self.biogeme_format.merge(path_share, left_on=['origin', 'destination', 'CHOICE', 'time_interval'],
                                         right_on= ['origin', 'destination', 'path_id',
                                              'time_interval'])  # assign the path share
        if len(demand) == 0:
            data['WEIGHT'] = data['path_share']
        else:
            demand['time_interval'] = demand['entry_interval'] // _DefaultValues.TIME_INTERVAL_CHOICE * _DefaultValues.TIME_INTERVAL_CHOICE
            demand = demand.groupby(['origin','destination','time_interval']).sum()[['flow']].reset_index()
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

        logprob = models.loglogit(V, av, CHOICE)#bioexp.bioLogLogit(V, av, CHOICE)
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
            data.to_csv(self.input_file_path + 'tb_path_share_synthetic.csv',index=False,columns=['origin', 'destination',\
                                                                                              'path_id', 'time_interval','path_share'])
        return data[['origin', 'destination', 'path_id', 'time_interval','path_share']]

    def generate_tb_exit_demand_synthesized(self, path_share, TEST):
        static = True
        # beta = [-0.0663, -0.438, -0.183,-0.941,-0.0767]
        beta = []
        OD_state, passenger_state, input_demand = _NPM_egine_path_choice.NPMModel(self.support_input, beta, path_share, static).run_assignment()
        OD_state = OD_state.loc[OD_state['flow'] > 0.1]
        if TEST == 1:
            OD_state.to_csv('Assignment_test/tb_exit_demand_synthesized2.csv', index=False)
        else:
            OD_state.to_csv(self.input_file_path + 'tb_exit_demand_synthesized2.csv', index=False)
            passenger_state.to_csv(self.input_file_path + 'tb_passenger_state_synthesized2.csv', index=False)

    def Calculate_CM_df(self, OD_state):
        OD_entry = self.input_demand
        OD_state['entry_interval'] = OD_state['entry_time'] // _DefaultValues.TIME_INTERVAL_DEMAND * _DefaultValues.TIME_INTERVAL_DEMAND
        OD_state['exit_interval'] = OD_state['exit_time'] // _DefaultValues.TIME_INTERVAL_DEMAND * _DefaultValues.TIME_INTERVAL_DEMAND
        OD_entry_exit = OD_state.groupby(['origin','destination','path','entry_interval','exit_interval']).sum()[['flow']].reset_index(drop=False)

        OD_entry_exit = OD_entry_exit.merge(OD_entry,left_on = ['origin','destination','path','entry_interval'],right_on \
            = ['origin','destination','path','entry_interval'])
        OD_entry_exit['contri_rate'] = OD_entry_exit['flow_x'] / OD_entry_exit['flow_y']
        OD_entry_exit = OD_entry_exit.merge(self.NODE,left_on = ['origin','entry_interval'], right_on = ['station','time_interval']).drop(columns =['station','time_interval'])
        OD_entry_exit = OD_entry_exit.merge(self.NODE, left_on=['destination', 'exit_interval'],
                            right_on=['station', 'time_interval']).drop(columns=['station', 'time_interval'])
        OD_entry_exit = OD_entry_exit.rename(columns = {'Node_index_x':'im','Node_index_y':'jn'})
        # only consider the opt time period
        OD_entry_exit = OD_entry_exit.loc[(OD_entry_exit['entry_interval'] >= self.opt_time_period[0]) & (OD_entry_exit['entry_interval'] < self.opt_time_period[1])]
        OD_entry_exit = OD_entry_exit.loc[(OD_entry_exit['exit_interval'] >= self.opt_time_period[0]) & (OD_entry_exit['exit_interval'] < self.opt_time_period[1])]
        filterstation = [76, 78, 57, 51, 52, 118]
        OD_entry_exit = OD_entry_exit.loc[(~OD_entry_exit['origin'].isin(filterstation)) & (~OD_entry_exit['destination'].isin(filterstation))]
        cm_df = OD_entry_exit.sort_values(['origin','destination','path','entry_interval'])
        # OD_entry = OD_entry.sort_values(['origin','destination','path','entry_interval'])

        return cm_df

    def generate_Estimated_Q(self, path_share_name):
        path_share = pd.read_csv(self.input_file_path + path_share_name)
        static = True
        beta = []
        OD_state, passenger_state, input_demand = _NPM_egine_path_choice.NPMModel(self.support_input, beta, path_share, static).run_assignment()
        # #--- Test for best scenarios----
        # OD_state = self.TB_EXIT_DEMAND_raw.copy()
        # passenger_state = pd.read_csv(self.input_file_path + 'tb_passenger_state_synthesized2.csv')
        #----------
        OD_state = OD_state.loc[OD_state['flow']>0.1]# filter the small (and negtive? precision?) flow to improve speed
        self.cm_df = self.Calculate_CM_df(OD_state)
        # # save MTR path share
        q_df_MTR = self.from_OD_state_to_q_df(OD_state)
        q_df_MTR = q_df_MTR.merge(self.q_true_df, left_on=['im','jn'], right_on=['im','jn'])
        q_df_MTR.to_csv('Estimated_Q'+path_share_name, index=False)


    def get_CM(self, path_share, last_CM_df, update_flag, time_period_list, info_station,iter):
        static = True
        beta = []
        OD_state, passenger_state, input_demand = _NPM_egine_path_choice.NPMModel(self.support_input, beta, path_share, static).run_assignment()
        # #--- Test for best scenarios----
        # OD_state.to_csv('test_OD_state.csv')
        # passenger_state.to_csv('test_passenger_state.csv')
        # a = aaaaaa
        # OD_state = self.TB_EXIT_DEMAND_raw.copy()
        # passenger_state = pd.read_csv(self.input_file_path + 'tb_passenger_state_synthesized2.csv')
        #----------
        q_df = self.from_OD_state_to_q_df(OD_state)
        objective_func = self.get_objective_function(q_df)


        OD_state = OD_state.loc[OD_state['flow']>0.1]# filter the small (and negtive? precision?) flow to improve speed
        self.cm_df = self.Calculate_CM_df(OD_state)

        if isinstance(update_flag, str):
            if iter != 0:
                self.cm_df = self.cm_df.merge(last_CM_df[['im', 'jn', 'path', 'contri_rate']], left_on=['im', 'jn', 'path'],
                                              right_on=['im', 'jn', 'path'], how='outer').fillna(0)
                if update_flag == 'Dynamic1':
                    CM_avg_factor_dynamic = 1 / (iter + 1)
                elif update_flag == 'Dynamic2':
                    CM_avg_factor_dynamic = 2/((iter + 1) + 1)

                self.cm_df['contri_rate'] = CM_avg_factor_dynamic * self.cm_df['contri_rate_x'] + (1 - CM_avg_factor_dynamic) * \
                                            self.cm_df['contri_rate_y']
                self.cm_df = self.cm_df.drop(columns=['contri_rate_x', 'contri_rate_y'])
        else:
            if update_flag == 1:
                self.cm_df = self.cm_df.merge(last_CM_df[['im','jn','path','contri_rate']], left_on = ['im','jn','path'],right_on = ['im','jn','path'], how = 'outer').fillna(0)
                self.cm_df['contri_rate'] = self.CM_avg_factor * self.cm_df['contri_rate_x'] + (1-self.CM_avg_factor)*self.cm_df['contri_rate_y']
                self.cm_df = self.cm_df.drop(columns = ['contri_rate_x','contri_rate_y'])

        jn_list = self.cm_df.groupby(['im','destination','path']).agg({'jn':lambda x: list(x)}) # any faster methods?
        OD_entry_exit = self.cm_df.set_index(['im', 'jn', 'path'])
        cm = OD_entry_exit.to_dict()['contri_rate']
        # print (cm[(14,326,1)])
        jn_list = jn_list.to_dict()['jn']
        # Calculate ADM left behind
        # passenger_state = passenger_state.loc[(passenger_state['departure_station'] == 2) &
        #                                       (passenger_state['departure_line'] == 11) &
        #                                       (passenger_state['departure_direction'] == 1)]  # filter out AEL

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

        return cm, self.cm_df, jn_list, LB, Arr, Board, objective_func

    def from_OD_state_to_q_df(self, OD_state):
        OD_state['entry_interval'] = OD_state['entry_time'] // _DefaultValues.TIME_INTERVAL_DEMAND * _DefaultValues.TIME_INTERVAL_DEMAND
        OD_state['exit_interval'] = OD_state['exit_time'] // _DefaultValues.TIME_INTERVAL_DEMAND * _DefaultValues.TIME_INTERVAL_DEMAND
        q_df = OD_state.groupby(['origin','destination','entry_interval','exit_interval']).sum()[['flow']].reset_index(drop=False)
        q_df = q_df.merge(self.NODE, left_on = ['origin','entry_interval'], right_on = ['station','time_interval']).drop(columns =['station','time_interval'])

        q_df = q_df.merge(self.NODE, left_on=['destination', 'exit_interval'],
                            right_on=['station', 'time_interval']).drop(columns=['station', 'time_interval'])
        q_df = q_df.rename(columns = {'Node_index_x':'im','Node_index_y':'jn'})
        q_df = q_df.sort_values(['origin', 'destination', 'entry_interval'])
        return q_df

    def data_format_transform(self):
        beta=[]
        static = True
        input_demand = _NPM_egine_path_choice.DemandModel(self.TB_TXN_RAW, choice_file=self.PATH_SHARE, beta=beta,
                                      ITINERARY=self.ITINERARY, ACC_EGR_TIME= self.ACC_EGR_TIME,
                                      static=static, TIME_INTERVAL_CHOICE = _DefaultValues.TIME_INTERVAL_CHOICE).tb_txn
        input_demand['entry_interval'] = input_demand['pax_tapin_time'] //\
                                         _DefaultValues.TIME_INTERVAL_DEMAND * _DefaultValues.TIME_INTERVAL_DEMAND
        input_demand = input_demand.loc[:,['pax_origin','pax_destination','entry_interval','pax_path','pax_number']].rename(
        columns = {'pax_origin':'origin','pax_destination':'destination','pax_path':'path','pax_number':'flow'})
        self.input_demand = input_demand.groupby(['origin', 'destination', 'path', 'entry_interval']).sum()[['flow']].reset_index()
        DEMAND = self.input_demand.groupby(['origin', 'destination', 'entry_interval']).sum()[['flow']].reset_index()
        # DEMAND.to_csv('demand_agg_flow.csv')
        self.entry_demand = DEMAND.merge(self.NODE, left_on=['origin','entry_interval'],right_on=['station','time_interval']) #
        # self.entry_demand.to_csv('entry_demand_agg.csv')
        self.entry_demand = self.entry_demand.set_index(['Node_index','destination'])[['flow']]

        self.entry_demand = self.entry_demand.to_dict()['flow']

        self.path_set = self.PATH_SHARE.loc[:, ['origin', 'destination', 'path_id']].drop_duplicates()
        self.path_set = self.path_set.groupby(['origin','destination']).apply(lambda x:list(x['path_id']))
        self.path_set = self.path_set.to_dict()
        #---------------
        q_true = self.from_OD_state_to_q_df(self.TB_EXIT_DEMAND_raw)
        self.q_true_df = q_true
        q_true = q_true.set_index(['im', 'jn'])
        self.q_true = q_true.to_dict()['flow']
        a=1


    def Solve_subp_1(self, cm, jn_list):
        m = _Subproblem1.OPT_Path_share(self.entry_demand, self.path_set, self.NODE, cm, jn_list,self.q_true, self.P_constraints, self.PATH_REFER)
        p_new, q_est, obj_value = m.main_model()
        return p_new, q_est, obj_value

    def From_list_to_df(self, p_new):
        p_new = pd.DataFrame.from_dict(p_new, orient='index', columns=['path_share']).reset_index(drop=False)
        p_new['path_share'] *= 100
        p_new[['origin', 'destination', 'path_id']] = pd.DataFrame(p_new['index'].tolist(),
                                                                   columns=['origin', 'destination', 'path_id'])
        p_new = p_new.drop(columns=['index'])
        return p_new

    def get_objective_function(self, q_df):
        # only consider the opt time period
        q_df = q_df.loc[(q_df['entry_interval'] >= self.opt_time_period[0]) & (q_df['entry_interval'] < self.opt_time_period[1])]
        q_df = q_df.loc[(q_df['exit_interval'] >= self.opt_time_period[0]) & (q_df['exit_interval'] < self.opt_time_period[1])]
        filterstation = [76, 78, 57, 51, 52, 118] # brunch stations
        q_df = q_df.loc[(~q_df['origin'].isin(filterstation)) & (~q_df['destination'].isin(filterstation))]
        q_df_compare = q_df.merge(self.q_true_df, left_on=['im','jn'], right_on=['im','jn'])
        obj = sum((q_df_compare['flow_x'] - q_df_compare['flow_y'])*(q_df_compare['flow_x'] - q_df_compare['flow_y']))
        return obj

    def excute(self, Synthetic_data, time_period_list, info_station, Intial_beta):
        # data process
        self.data_format_transform()
        self.generate_Biogeme_files()


        #----------------
        # Run the moodel
        # only two beta: in-veh time, No of transfer, because only these two are stable

        pathshare = self.beta_to_path_share(Intial_beta, save= 0)
        # Calculate initial RMSE
        p_new = self.PATH_SHARE[['origin', 'destination', 'path_id', 'time_interval', 'path_share']]. \
            merge(pathshare, left_on=['origin', 'destination', 'path_id','time_interval'],
                  right_on=['origin', 'destination', 'path_id','time_interval'], how='left').fillna(0)
        p_new = p_new.rename(columns={'path_share_x': 'path_share_true', 'path_share_y': 'path_share'})
        p_new['error'] = (p_new['path_share_true'] - p_new['path_share']) ** 2
        RMSE = np.sqrt(p_new['error'].mean())
        # save the initial p_share, which should be the worst estimates we can got (baseline)
        # p_new.to_csv('Estimated_P.csv', index=False,
        #              columns=['origin', 'destination', 'path_id', 'time_interval', 'path_share_true', 'path_share'])
        beta = self.path_share_to_beta(p_new.drop(columns = ['error','path_share_true']), self.q_true_df)
        df_dict = {'Iter': [0], 'RMSE': [RMSE], 'Obj_value_before': [-1],'Obj_value_after': [-1]}
        for info_time_interval in time_period_list:
            for info_s in info_station:
                name_LB = 'LB_' + str(info_s[0]) + '_' + str(info_s[1]) + '_' + str(info_s[2]) + '_' +  str(info_time_interval)
                name_Arr = 'Arr_' + str(info_s[0]) + '_' + str(info_s[1]) + '_' + str(info_s[2]) + '_' +  str(info_time_interval)
                name_Board = 'Board_' + str(info_s[0]) + '_' + str(info_s[1]) + '_' + str(info_s[2]) + '_' +  str(info_time_interval)
                df_dict[name_LB] = [-1]
                df_dict[name_Arr] = [-1]
                df_dict[name_Board] = [-1]

        records = pd.DataFrame(df_dict)
        for key in beta:
            beta[key] = [beta[key]]
        records = pd.concat([records, pd.DataFrame(beta)], axis=1)  # concat on columns

        last_CM_df = pd.DataFrame()
        # Name the file
        tail_name = self.tail_name
        record_name = 'Model_records_'+ tail_name +'.csv'
        P_name = 'Estimated_P_'+ tail_name +'.csv'
        Q_name = 'Estimated_Q_'+ tail_name +'.csv'
        CM_name = 'CM_'+ tail_name +'.csv'
        if Synthetic_data == 1:
            print('Intial RMSE between synthetic:', RMSE)
        else:
            print('Intial RMSE between MTR:', RMSE)
        print('Intial beta', beta)

        # Test for best eatimates
        # pathshare = self.PATH_SHARE
        # number_iter = 1

        # Choose for number of iteration
        number_iter = 50

        # ------
        iter_began_avg = 0
        for iter in range(number_iter): #
            print('------------Iter',iter,'---------------')
            if isinstance(self.CM_avg_factor, str):
                update_flag = self.CM_avg_factor
                cm, cm_df, jn_list, left_behind, arr, Board, objective_func= self.get_CM(pathshare, last_CM_df,
                                                                   update_flag, time_period_list,
                                                                   info_station,iter)  # CM: contribution matrix
            else:
                if iter <= iter_began_avg:
                    update_flag = 0
                    cm, cm_df, jn_list, left_behind, arr, Board,objective_func = self.get_CM(pathshare, last_CM_df,
                                                                       update_flag, time_period_list, info_station,iter)  # CM: contribution matrix
                else:
                    update_flag = 1
                    cm, cm_df, jn_list, left_behind, arr, Board,objective_func = self.get_CM(pathshare, last_CM_df,
                                                                       update_flag, time_period_list, info_station,iter)  # CM: contribution matrix
            last_CM_df = cm_df
            p_new, q_est, obj_value = self.Solve_subp_1(cm, jn_list)
            # transfer to df
            p_new = self.From_list_to_df(p_new)
            # Add time interval
            p_new_raw = self.PATH_SHARE[['origin', 'destination', 'path_id','time_interval']]. \
                merge(p_new, left_on=['origin', 'destination', 'path_id'],
                      right_on=['origin', 'destination', 'path_id'], how='left').fillna(0)
            # look at the row path share
            # p_new_raw = self.PATH_SHARE[['origin', 'destination', 'path_id', 'time_interval', 'path_share']]. \
            #     merge(p_new_raw, left_on=['origin', 'destination', 'path_id','time_interval'],
            #           right_on=['origin', 'destination', 'path_id','time_interval'], how='left').fillna(0)
            # p_new_raw = p_new_raw.rename(columns={'path_share_x': 'path_share_true', 'path_share_y': 'path_share'})
            # p_new_raw.to_csv(P_name.replace(".csv", "_raw.csv"), index=False,
            #              columns=['origin', 'destination', 'path_id', 'time_interval', 'path_share_true', 'path_share'])
            # estimate beta
            beta = self.path_share_to_beta(p_new_raw, self.q_true_df)
            # Use estimated beta to generate new path share (correct 0 and low-demand estimates)
            pathshare = self.beta_to_path_share(beta, save=0)
            # # Calculate RMSE
            # pathshare = p_new
            p_new = self.PATH_SHARE[['origin', 'destination', 'path_id', 'time_interval', 'path_share']]. \
                merge(pathshare, left_on=['origin', 'destination', 'path_id','time_interval'],
                      right_on=['origin', 'destination', 'path_id','time_interval'], how='left').fillna(0)
            p_new = p_new.rename(columns={'path_share_x': 'path_share_true', 'path_share_y': 'path_share'})

            #---
            p_new['error'] = (p_new['path_share_true'] - p_new['path_share']) ** 2
            RMSE = np.sqrt(p_new['error'].mean())
            if Synthetic_data == 1:
                print ('Iter' + str(iter), 'RMSE between synthetic:', RMSE)
            else:
                print('Iter' + str(iter), 'RMSE bwteen MTR:', RMSE)
            print ('Iter' + str(iter), 'Subprob 2 obj_value before:', objective_func)
            print( 'Iter' + str(iter), 'Subprob 2 obj_value after:', obj_value)
            print ('Iter' + str(iter), 'beta:', beta)

            df_dict = {'Iter': [iter+1], 'RMSE': [RMSE],'Obj_value_before': [objective_func], 'Obj_value_after': [obj_value]}
            for info_time_interval in time_period_list:
                for info_s in info_station:
                    name_LB = 'LB_' + str(info_s[0]) + '_' + str(info_s[1]) + '_' + str(info_s[2]) + '_' +  str(
                        info_time_interval)
                    name_Arr = 'Arr_' + str(info_s[0]) + '_' + str(info_s[1]) + '_' + str(info_s[2]) + '_' +  str(
                        info_time_interval)
                    name_Board = 'Board_' + str(info_s[0]) + '_' + str(info_s[1]) + '_' + str(info_s[2]) + '_' + str(
                        info_time_interval)
                    df_dict[name_LB] = [left_behind[(info_time_interval, info_s)]]
                    df_dict[name_Arr] = [arr[(info_time_interval, info_s)]]
                    df_dict[name_Board] = [Board[(info_time_interval, info_s)]]


            temp_records = pd.DataFrame(df_dict)
            for key in beta:
                beta[key] = [beta[key]]
            temp_records = pd.concat([temp_records, pd.DataFrame(beta)], axis=1)  # concat on columns
            records = records.append(temp_records)
            # save in every iteration
            records.to_csv(record_name, index=False)
            # print(p_new)

        # save final p_new
        p_new.to_csv(P_name, index=False,
                     columns=['origin', 'destination', 'path_id', 'time_interval', 'path_share_true', 'path_share'])
        # save final q_est
        q_est = pd.DataFrame.from_dict(q_est, orient = 'index',columns = ['flow']).reset_index(drop=False)
        q_est[['im','jn']] = pd.DataFrame(q_est['index'].tolist(), columns=['im','jn'])
        q_est = q_est.drop(columns=['index'])
        q_est = q_est.merge(self.q_true_df, left_on=['im','jn'], right_on=['im','jn'])
        q_est.to_csv(Q_name, index=False)
        # save final CM
        # CM_name = CM_name
        # self.cm_df = self.cm_df.sort_values(['origin','destination','path','entry_interval'])
        # cm_df_true = self.Calculate_CM_df(self.TB_EXIT_DEMAND_raw)
        # cm_df_true = cm_df_true.rename(columns = {'contri_rate':'contri_rate_true'})
        # self.cm_df = self.cm_df.merge(cm_df_true[['im','jn','path','contri_rate_true']],left_on = ['im','jn','path'], right_on = ['im','jn','path'],how = 'outer')
        # self.cm_df.to_csv(CM_name, index=False)
        # save records
        records.to_csv(record_name, index=False)
# -----------------------------------------  Main Function ---------------------------------------------------

if __name__ == "__main__":

    rand_seed_list = [0]
    for rand_seed_start in rand_seed_list:
        #os.chdir('F:\[Network Performance Model]\Code') #change current file path
        TEST = 0 #1 means run with test data 0 means not
        name = '_syn'
        date = '2017-03-16'

        if TEST == 1:
            para_list = pd.read_csv('0_user_configuration_parameter_TEST.csv').dropna()
            info_station = []
        else:
            para_list = pd.read_csv('0_user_configuration_parameter.csv').dropna()
            info_station = [(2, 11, 1)]  # station + line + dir
        tic=time.time()
        for ix, para in para_list.iterrows():
            Synthetic_data = 1 # 1 = test with synthetic data
            generate_demand = False # True if we only generate synthetic data not runing the model
            generate_beta = False
            time_interval_demand = 15*60 # resolution of exit OD
            sim_time_period = para['sim_time_period'].split('-')
            time_start = int(pd.to_timedelta(sim_time_period[0]).total_seconds())
            time_end = int(pd.to_timedelta(sim_time_period[1]).total_seconds())
            time_period_list = list(range(time_start, time_end, time_interval_demand))

            CM_avg_factor = 1 # or number 0-1. Number 0-1 means fixed
            # CM_avg_factor = 1 #(1-CM_avg_factor)*last_CM + CM_avg_factor*new_CM # = 1 means no avg
            file_tail_name = date + name + '_' + str(CM_avg_factor) + '_' + str(rand_seed_start)
            path_att = _DefaultValues.path_att # names need to consistent with path attributes file

            # Intial_beta = {'B_IVT': -0.0663, 'B_NoTr': -0.438}
            # Intial_beta =  {'B_IVT':-0.1, 'B_NoTr':-1}
            beta_bound_dict = {'in_vehicle_time': [-2, 0], 'no_of_transfer': [-4, 0], 'transfer_over_dist': [-6, 0],
                               'commonality_factor': [-10, 0]}
            Intial_beta = {}
            # rand_seed = rand_seed_start * 1000 + 20 #
            # for att_name in path_att:
            #     rand_seed += 1
            #     np.random.seed(rand_seed)
            #     ini_value = np.random.uniform(beta_bound_dict[att_name][0], beta_bound_dict[att_name][1])
            #     beta_name = 'B_' + str.upper(att_name)
            #     Intial_beta[beta_name] = ini_value
            Intial_beta = {'B_IN_VEHICLE_TIME': 0, 'B_NO_OF_TRANSFER': 0, 'B_TRANSFER_OVER_DIST': 0,
             'B_COMMONALITY_FACTOR': 0}

            # ===============
            my_opt = OptimizationModel(para, time_interval_demand, CM_avg_factor, path_att, TEST,file_tail_name, date, name)
            my_opt.Load_input_files(TEST = TEST, Synthetic_data = Synthetic_data, generate_data= generate_demand)
            my_opt.Load_input_files_SIM(TEST = TEST, Synthetic_data = Synthetic_data, generate_data= generate_beta)
            print('load file time:', time.time() - tic)

            if generate_demand:
                print('--------Generate data--------')
                #-----Generate new synthetic demand
                # my_opt.generate_tb_exit_demand_synthesized(my_opt.PATH_SHARE, TEST = TEST)
            elif generate_beta:
                print('--------Generate data--------')
                #-----Generate new path share
                # beta = {'B_IN_VEHICLE_TIME':-0.0663, 'B_NO_OF_TRANSFER':-0.438,'B_TRANSFER_TIME':-0.183, 'B_COMMONALITY_FACTOR':-0.941}
                # my_opt.beta_to_path_share(beta, save= 1)
                # beta = {'B_IN_VEHICLE_TIME':-0.147, 'B_NO_OF_TRANSFER':-0.573,'B_TRANSFER_OVER_DIST':-1.271, 'B_COMMONALITY_FACTOR':-3.679}
                # my_opt.beta_to_path_share(beta, save= 1)
            else:
                #-----Test beta estamatability
                # path_share_to_generate_beta = pd.read_csv(my_opt.input_file_path + 'tb_path_share_synthetic.csv')
                # my_opt.data_format_transform()
                # my_opt.generate_Biogeme_files()
                # beta = my_opt.path_share_to_beta(path_share_to_generate_beta, my_opt.q_true_df)
                # print(beta)
                #----Test demand stability
                # path_file_name = 'tb_path_share_synthetic.csv'
                # my_opt.data_format_transform()
                # my_opt.generate_Estimated_Q(path_file_name)

                # print('--------Excute model--------')
                print('file_name:',file_tail_name)
                print('Synthetic data:',Synthetic_data)
                print('CM avg factor:', CM_avg_factor)
                print('path_att:',path_att)
                my_opt.excute(Synthetic_data, time_period_list, info_station, Intial_beta)


        # tapin_csv_file.close()
        print('Total time:', time.time() - tic, 'sec')
    #x = input('Press any key to exit ...')