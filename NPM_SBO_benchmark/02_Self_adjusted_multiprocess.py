
##############################################################################################
# This program is used to assign OD demand on network with link capacity constraints and flow priority
# Inputs: Passenger graph and set of passenger groups
# Output: Flow function
# Note: 1. the time is recorded as the minutes difference compared to a reference time (e.g. 7:00 am).
#       2. OD demand is time, demand in pandas.
#       3. Dictionaries in Python are implemented as hash tables, there is no ordering. To track the order of joined
#          queue with arrival times, it needs a list of order of the arrival times.
#       4. tap_in demand is in time period, for example 7:15 = 7:00-7:15


import _blackbox
from SPSA import SimpleSPSA
import sys
sys.path.insert(0,'..')
import _NPM_egine_path_choice
import _DefaultValues
import pandas as pd
import os
import numpy as np
import sys
import copy
import time
import random
import scipy
import math
import multiprocessing as mp

from bayes_opt import BayesianOptimization
from bayes_opt.util import UtilityFunction

pd.options.mode.chained_assignment = None  # default='warn'



class OptimizationModel(object):
    def __init__(self, para_npm, time_interval_demand,used_att,TEST, file_tail_name, date, name):
        self.para = para_npm
        self.DATE = date
        self.iter = 0
        self.outputname = '.csv'
        sim_time_period = para_npm['sim_time_period'].split('-')
        time_start = int(pd.to_timedelta(sim_time_period[0]).total_seconds())
        time_end = int(pd.to_timedelta(sim_time_period[1]).total_seconds())
        self.opt_time_period = [time_start, time_end]
        self.PATH_ATTRIBUTE = pd.DataFrame()
        self.NODE = pd.DataFrame()
        self.DEMAND = pd.DataFrame()
        self.time_interval_demand = time_interval_demand
        self.time_interval = time_interval_demand
        self.input_file_path = '../Assignment_' + str(self.opt_time_period[0]) + '-' + str(
            self.opt_time_period[1]) + '_' +date + name + '/'
        self.biogeme_format = pd.DataFrame()
        self.used_att = used_att
        self.Max_path_num = -1 #intialize
        self.tail_name = file_tail_name
        self.results = {'Iter':[],'RMSE':[],'Obj':[]}
        for key in self.used_att:
            beta_name = 'B_' + str.upper(key)
            self.results[beta_name] = []



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
                self.TB_EXIT_DEMAND = self.process_OD_state_to_demand(self.TB_EXIT_DEMAND_raw)
            else:
                self.TB_EXIT_DEMAND_raw = pd.read_csv(input_file_path + 'tb_txn.csv')
                # process for the real-world test
                self.TB_EXIT_DEMAND_raw = self.TB_EXIT_DEMAND_raw.rename(columns = {'pax_tapin_time':'entry_time',\
                'pax_tapout_time':'exit_time', 'pax_origin':'origin', 'pax_destination':'destination'})
                self.TB_EXIT_DEMAND_raw['flow'] = 1
                self.TB_EXIT_DEMAND_raw = self.TB_EXIT_DEMAND_raw.drop(columns= ['user_id','tapout_ti'])



    def Load_input_files_SIM(self, TEST, Synthetic_data, generate_data):

        # Variable settings
        path_external_data = '../External_data/'  # input data are in External_data folder
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

    def process_OD_state_to_demand(self,OD_state):

        OD_state['entry_time_interval'] = OD_state['entry_time'] // self.time_interval * self.time_interval
        OD_state['exit_time_interval'] = OD_state['exit_time'] // self.time_interval * self.time_interval
        OD_state = OD_state.loc[(OD_state.entry_time_interval >= self.opt_time_period[0]) & (OD_state.entry_time_interval <= self.opt_time_period[1])]
        OD_state = OD_state.loc[(OD_state.exit_time_interval >= self.opt_time_period[0]) & (
                    OD_state.exit_time_interval <= self.opt_time_period[1])]
        filterstation = [76, 78, 57, 51, 52, 118]
        OD_state = OD_state.loc[(~OD_state['origin'].isin(filterstation)) & (~OD_state['destination'].isin(filterstation))]
        est_flow = OD_state.groupby([OD_state.origin, OD_state.destination,
                                     OD_state.entry_time_interval,OD_state.exit_time_interval])['flow'].sum().reset_index(drop=False)
        return est_flow

    def beta_to_path_share(self, beta, save = 0):
        beta_dict = {}
        idx = 0
        for key in self.used_att:
            beta_name = 'B_' + str.upper(key)
            beta_dict[beta_name] = beta[idx]
            idx += 1
        data = self.PATH_ATTRIBUTE.copy()
        data['Utility_exp'] = 0
        for name in self.used_att:
            beta_name = 'B_' + str.upper(name)
            data['Utility_exp'] += data[name] * beta_dict[beta_name]
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

    def calculate_RMSE_of_path_share(self,path_share):
        p_new = self.PATH_SHARE[['origin', 'destination', 'path_id', 'time_interval', 'path_share']]. \
            merge(path_share, left_on=['origin', 'destination', 'path_id','time_interval'],
                  right_on=['origin', 'destination', 'path_id','time_interval'], how='left').fillna(0)
        p_new = p_new.rename(columns={'path_share_x': 'path_share_true', 'path_share_y': 'path_share'})
        p_new['error'] = (p_new['path_share_true'] - p_new['path_share']) ** 2
        RMSE = p_new['error'].mean()
        return RMSE

    def BlackboxFunc(self, beta):
        print('current beta is:', beta)
        path_share = self.beta_to_path_share(beta, save= 0)
        static = True
        cap_para = [0,0]
        beta_empty = []
        OD_state, passenger_state, input_demand = _NPM_egine_path_choice.NPMModel(self.support_input, beta_empty, path_share, static).run_assignment()

        #OD_state.to_csv('NPM_egine_true_OD_state_ForFLOW.csv',index=False)

        # passenger_state.to_csv('NPM_egine_true_beta_passenger_state_state.csv',index=False)
        # Calculate the estimated OD exit flow matrix
        w1 = 1
        #w2 = 70
        exit_flow_diff = self.BlackboxFunc_15min_OD(OD_state)
        print('current obj_function is:', exit_flow_diff)

        # self.iter += 1
        # idx = 0
        # for key in self.used_att:
        #     beta_name = 'B_' + str.upper(key)
        #     self.results[beta_name].append(beta[idx])
        #     idx += 1
        # self.results['Obj'].append(exit_flow_diff)
        # rmse = self.calculate_RMSE_of_path_share(path_share)
        # self.results['RMSE'].append(rmse)
        # self.results['Iter'].append(self.iter)
        #entropy = self.BlackboxFunc_Entropy(OD_state)
        # temp_output = pd.DataFrame(self.results)
        # temp_output.to_csv('Model_records_' + self.outputname)
        return w1*exit_flow_diff #+ w2*entropy

    def get_OD_state(self,beta):
        path_share = self.beta_to_path_share(beta, save= 0)
        static = True
        cap_para = [0,0]
        beta = []
        OD_state, passenger_state, input_demand = _NPM_egine_path_choice.NPMModel(self.support_input, beta, path_share, static).run_assignment()

        return OD_state

    # def BlackboxFunc_Entropy(self, OD_state):
    #     TB_EXIT_DEMAND_Entropy = self.TB_EXIT_DEMAND_raw.groupby([self.TB_EXIT_DEMAND_raw.origin, self.TB_EXIT_DEMAND_raw.destination,
    #                                                        self.TB_EXIT_DEMAND_raw.time]).filter(lambda x: len(x)>50) # MORE THAN 50 samples
    #     TB_EXIT_DEMAND_Entropy['travel_time'] = TB_EXIT_DEMAND_Entropy['exit_time'] - TB_EXIT_DEMAND_Entropy['entry_time']
    #     TB_EXIT_DEMAND_Entropy = TB_EXIT_DEMAND_Entropy.groupby(['origin','destination','time']).\
    #         apply(lambda x: np.histogram(x['travel_time'], bins = 7, density=False)[0]/len(x)).reset_index().rename(columns = {0:'tt_bin'}) # 7 different time invertals
    #
    #     OD_state = OD_state.groupby([OD_state.origin, OD_state.destination,
    #                                                        OD_state.time]).filter(lambda x: len(x)>50) # MORE THAN 50 samples
    #     OD_state['travel_time'] = OD_state['exit_time'] - OD_state['entry_time']
    #     OD_state = OD_state.groupby(['origin','destination','time']).\
    #         apply(lambda x: np.histogram(x['travel_time'], bins = 7, density=False)[0]/len(x)).reset_index().rename(columns = {0:'tt_bin'}) # 7 different time invertals
    #     if len(OD_state) > 0:
    #         # Merge the est flow with exit demand data
    #         flow_merge = OD_state.merge(TB_EXIT_DEMAND_Entropy, on=['origin', 'destination', 'time'], how='inner')
    #         flow_merge['KL_div'] = -99
    #         for index, row in flow_merge.iterrows():
    #             if len(row['tt_bin_x']) == len(row['tt_bin_y']):
    #                 flow_merge.loc[index,'KL_div'] = scipy.stats.entropy(row['tt_bin_x'],row['tt_bin_y'])
    #         #flow_merge.to_csv('Entropy_record.csv', index=False)
    #         flow_merge = flow_merge.loc[(flow_merge['KL_div'] != -99)&
    #                                     (flow_merge['KL_div'] != np.inf)]
    #         travel_time_distribution_entropy = flow_merge['KL_div'].sum()
    #         print ('KL_div is', travel_time_distribution_entropy)
    #         return travel_time_distribution_entropy
    #     else:
    #         print('Error occurs is KL div calculation ...')
    #         return np.inf


    def BlackboxFunc_15min_OD(self, OD_state):
        """ Objective using exit demand flow seems to be not very sensitive
        Other options: K-L divergence of journey time distribution for Origin_Destination_Time
        """
        # Run the NPM model and get the OD_state
        # Make the inputs consistent with different optimization solver

        est_flow = self.process_OD_state_to_demand(OD_state)
        # print (OD_state)

        # print (est_flow)
        # Only keep the record within specified optimization time period

        if len(est_flow) > 0:
            # Merge the est flow with exit demand data
            est_act_exit_flow = est_flow.merge(self.TB_EXIT_DEMAND, on=['origin', 'destination', 'entry_time_interval','exit_time_interval'], how='inner')

            # Calculate the difference between estimated flow (from npm model) and the actual (from afc and sjc data)
            est_act_exit_flow['diff'] = est_act_exit_flow['flow_x'] - est_act_exit_flow['flow_y']
            est_act_exit_flow['squared_diff'] = est_act_exit_flow['diff'] ** 2

            # the objective value is the squared root of the estimated and actual exit flow difference
            # obj = (est_act_exit_flow['squared_diff'].sum() / len(est_act_exit_flow)) ** 0.5
            print ('num of od pair:',len(est_act_exit_flow))
            obj = est_act_exit_flow['squared_diff'].sum()

            # print('objective function value is %s' % obj)
            # The Bayesian optimizer MAXIMIZE the objective function
            print('OD diff is', obj)
            return obj
        else:
            print('Error occurs in OD diff calculation ...')
            return np.inf



class BayesianOptSolver:
    def __init__(self, num_of_parallel, num_of_iter, opt_model,bound,output_file_name,seed):
        self.out_put = pd.DataFrame()
        self.num_of_parallel = num_of_parallel
        self.num_of_iter = num_of_iter
        self.opt_model = opt_model
        self._bo = None
        self._uf = None
        self.output_file_name = output_file_name
        self.bound = bound
        self.seed = seed
        self.diff_bound = [0.05*abs(item[0] - item[1]) for item in bound]# each variable should diff larger than bound for different iterations

    def black_box_function(self, beta):
        # beta = [x1, x2, x3, x4, x5, x6]
        obj = self.opt_model.BlackboxFunc(beta)
        # Bayesian, default is max
        return float(-obj)

    def model_specification(self):
        self._bo = BayesianOptimization(
            f=self.black_box_function,
            pbounds={"x1": tuple(self.bound[0]), "x2": tuple(self.bound[1]), "x3":tuple(self.bound[2]),
         "x4": tuple(self.bound[3])},
            random_state= self.seed
        )
        # The parameters name should be able to sort!! with the sample sequence of beta list.
        '''
            x1 * self.tb_path_attribute['in_vehicle_time'] +
            x2 * self.tb_path_attribute['no_of_transfer'] +
            x3 * self.tb_path_attribute['transfer_time'] +
            x4 * self.tb_path_attribute['commonality_factor'] +
        '''
        self._uf = UtilityFunction(kind="ei", kappa=1, xi=1)
    def get_random_next(self):
        bound_sort = self.bound
        return [random.uniform(bound[0], bound[1]) for bound in bound_sort]
    def run_optimizer(self):
        pool =  mp.Pool()
        next_point = []
        next_point_to_register_old = []
        next_point_to_register_new = []
        EXISTING_POINTS = []
        output_file_path = self.output_file_name
        if os.path.exists(output_file_path):
            self.out_put = pd.read_csv(output_file_path).dropna()
            if len(self.out_put)>0:
                print('prior register')
                for idx, info in self.out_put.iterrows():
                    old_para = {"x1": info['x1'], "x2": info['x2'], "x3": info['x3'],
                                "x4": info['x4']}
                    EXISTING_POINTS.append([info['x1'], info['x2'], info['x3'],
                                            info['x4']])
                    print(old_para, info['target'])
                    self._bo.register(params=old_para, target=info['target'])


        target = []
        for _ in range(self.num_of_iter):
            time_now = time.time()
            for i in range(self.num_of_parallel):
                #==============Register=============
                if len(next_point_to_register_old) != 0:
                    self._bo.register(params=next_point_to_register_old[i], target=target[i])
                    new_output = pd.DataFrame(
                            {'x1': [next_point_to_register_old[i]['x1']], 'x2': [next_point_to_register_old[i]['x2']],
                             'x3': [next_point_to_register_old[i]['x3']],
                             'x4': [next_point_to_register_old[i]['x4']],
                             'target': [target[i]]})
                    if len(self.out_put) == 0:
                        self.out_put = copy.deepcopy(new_output)
                    else:
                        self.out_put = pd.concat([self.out_put, new_output])
                else:
                    pass
                #==============Get Next=============
                beta_dict = self._bo.suggest(self._uf)

                beta_dict_items = sorted(beta_dict.items(), key=lambda kv: kv[0])
                new_sample = [item[1] for item in beta_dict_items]
                # check if there is duplicated probed
                for node in EXISTING_POINTS:
                    check = [abs(new_sample[k] - node[k]) for k in range(len(node))]
                    check_bool = [check[j]>self.diff_bound[j] for j in range(len(check))]
                    if True in check_bool:
                        continue
                    else: # all false, there is a duplicated one
                        print ('Duplicated point, change one to random draw')
                        new_sample = self.get_random_next()
                        beta_dict = {'x1':new_sample[0],'x2':new_sample[1],'x3':new_sample[2],
                                     'x4':new_sample[3]}
                        break
                next_point.append(new_sample)
                EXISTING_POINTS.append(new_sample)
                next_point_to_register_new.append(beta_dict)
            # ======Parallel====
            print("BO has registered: {} points.".format(len(self._bo.space)))
            print('Total register time:', round((time.time() - time_now) / 60, 2), 'min', end="\n\n")
            self.out_put.to_csv(output_file_path, index=False) # output process
            print('next point:',next_point)
            target =  list(pool.map(self.black_box_function,next_point))
            # ======
            next_point_to_register_old = copy.deepcopy(next_point_to_register_new)
            next_point_to_register_new = []
            next_point = []
            print('Current_best',sorted(self._bo.max.items(), key=lambda kv: kv[0]))


        print("Optimization is done!", end="\n\n")
    def execute(self):
        #==== Model specification
        self.model_specification()
        #==== Run model
        self.run_optimizer()
        #====
        return self.out_put


class SPSA_solver: # minimize
    # Correct Gap by TT, using FDSA
    # Funtion ObjFunc
    def __init__(self, beta_ini, num_of_parallel, num_of_iter, opt_model,bound, noise_var,a_par,output_name,random_seed):
        self.out_put = pd.DataFrame()
        #self.num_of_parallel = num_of_parallel
        self.num_of_iter = num_of_iter
        self.num_of_parallel = num_of_parallel
        self.opt_model = opt_model
        self.bound = bound
        self.a_par = a_par
        self.results = pd.DataFrame()
        self.beta_ini = np.array(beta_ini)
        self.noise_var = noise_var
        self.ouput_name = output_name
        self.random_seed = random_seed

    def ObjFunc(self, beta):
        #obj =  -beta[0] * beta[0] + beta[1] * beta[1] + beta[2] * beta[2]
        obj = self.opt_model.BlackboxFunc(beta)
        new_output = pd.DataFrame(
            {'x1': [beta[0]], 'x2': [beta[1]],
             'x3': [beta[2]],
             'x4': [beta[3]], 'target': [obj]})

        self.results = pd.concat([self.results, new_output],sort=False)
        self.results.to_csv(self.ouput_name)
        return float(obj)


    def excute(self):
        opti = SimpleSPSA(self.ObjFunc,noise_var=self.noise_var,a_par = self.a_par, min_vals=[bd[0] for bd in self.bound],
                          max_vals=[bd[1] for bd in self.bound], max_iter = self.num_of_iter,n_jobs = self.num_of_parallel,random_seed=self.random_seed)
        (xsol, j_opt, niter) = opti.minimise(theta_0 = self.beta_ini, report=5)
        print('final results', 'x:', xsol, 'obj:',j_opt, 'iter:',niter)


class NelderMead: # minimize

    def __init__(self, beta_ini, num_of_iter, opt_model,bound,output_name):
        self.out_put = pd.DataFrame()
        #self.num_of_parallel = num_of_parallel
        self.num_of_iter = num_of_iter
        self.opt_model = opt_model
        self.bound = bound
        self.results = pd.DataFrame()
        self.beta_ini = np.array(beta_ini)
        self.output_name = output_name


    def ObjFunc(self, beta):
        #obj =  -beta[0] * beta[0] + beta[1] * beta[1] + beta[2] * beta[2]
        obj = self.opt_model.BlackboxFunc(beta)
        new_output = pd.DataFrame(
            {'x1': [beta[0]], 'x2': [beta[1]],
             'x3': [beta[2]],
             'x4': [beta[3]], 'target': [obj]})

        self.results = pd.concat([self.results, new_output],sort=False)
        self.results.to_csv(self.output_name,encoding='utf-8', index=True)
        return float(obj)


    def execute(self):
        opt = {'maxfev':self.num_of_iter,'disp': True}
        bound = [(bd[0],bd[1]) for bd in self.bound]
        result = scipy.optimize.minimize(self.ObjFunc, self.beta_ini, method='Nelder-Mead', bounds=self.bound, options=opt)
        print('final results', 'x:', result.x, 'obj:',result.fun, 'iter:',result.nit)

def CORS_RBF_solver(opt_model,bound,random_seed,output_name, num_LT_search, num_follow_search,num_of_parallel):
    # https://github.com/paulknysh/blackbox
    _blackbox.search(f=opt_model.BlackboxFunc,  # given function
              box= bound, # range of values for each parameter (2D case)
              n=int(num_LT_search),  # number of function calls on initial stage (global search)
              m=int(num_follow_search),  # number of function calls on subsequent stage (local search)
              batch=int(num_of_parallel),  # number of calls that will be evaluated in parallel
              resfile = output_name.replace('.csv','_sorted.csv'), # text file where results will be saved
              resfile_not_sort = output_name,
              random_seed=random_seed)  # text file where results will be saved 

def get_random_beta(bound):
    bound_sort = bound
    return [random.uniform(bd[0], bd[1]) for bd in bound_sort]

def test_for_beta(opt_model, beta_list):
    record = pd.DataFrame()
    for beta in beta_list:
        obj = opt_model.BlackboxFunc(beta)
        new_output = pd.DataFrame(
            {'x1': [beta[0]], 'x2': [beta[1]],
             'x3': [beta[2]],
             'x4': [beta[3]], 'target': [obj]})
        record = pd.concat([record,new_output], ignore_index=True)
        print ('beta is:',beta)
        print ('objective function is:', obj)
    record.to_csv('Beta_test_results.csv',index=False)


def Generate_synthetic_demand(opt_model, beta_list):
    OD_state = opt_model.get_OD_state(beta_list)
    OD_state.to_csv(opt_model.input_file_path+'tb_exit_demand_synthesized.csv',index=False)

class Nomad_solver:
    # rely on "import PyNomad", refer to https://www.gerad.ca/nomad/ about how to install it (or look at my slides guidance)
    # Baichuan 2019/2/26
    def __init__(self, num_of_parallel, num_of_iter, opt_model,bound,intial_beta,output_name):

        self.num_of_parallel = num_of_parallel
        self.num_of_iter = num_of_iter
        self.opt_model = opt_model
        self.bound = bound
        self.intial_beta = intial_beta
        self.results = pd.DataFrame()
        self.output_name = output_name

    def black_box_function(self, beta):
        obj = self.opt_model.BlackboxFunc(beta)
        # Nomad, default is min
        #obj = -beta[0] * beta[0] + beta[1] * beta[1] + beta[2] * beta[2]

        return float(obj)

    def input_bb(self, x, bb_out):
        # ---------------------------------------------------------------------
        # THIS EXAMPLES DOES NOT WORK ON DEFAULT WINDOWS ANACONDA INSTALLATION
        # ---------------------------------------------------------------------

        # This example of blackbox function is for multiprocess: several bb are called in parallel
        # Multiprocess blackbox evaluations request to provide the BB_MAX_BLOCK_SIZE in NOMAD params
        #
        # In this case blackbox function requires 2 arguments
        # The first argument x is similar to a NOMAD::Eval_Point --> access the coordinates with get_coord() function
        # The blackbox output must be put in the second argument bb_out and the bb outputs must be put in the same order as defined in params
        # bb_out is a queue to work in a multiprocess.
        # try:
        dim = x.get_n()
        beta = [x.get_coord(i) for i in range(dim)]
        # print(self.out_put)
        f = self.black_box_function(beta) # f is the returned obj
        #----save----
        # Nomad use different cores, we have to save it on the disk and read resulst in every iteration
        out_file_name = self.output_name
        if not os.path.exists(out_file_name):
            Results = pd.DataFrame({'x1': [], 'x2':[],
                                    'x3':[], 'x4':[],'target':[]})
            Results.to_csv(out_file_name, index=False)

        new_output = pd.DataFrame(
            {'x1': [beta[0]], 'x2': [beta[1]],
             'x3': [beta[2]],
             'x4': [beta[3]], 'target': [f]})
        new_output.to_csv(out_file_name, index=False, mode='a',header  = False)
        #----
        # gi is the contraints
        # g1 = sum([(x.get_coord(i) - 1) ** 2 for i in range(dim)]) - 25
        # g2 = 25 - sum([(x.get_coord(i) + 1) ** 2 for i in range(dim)])
        bb_out.put([f], block = False)
        # except:
        #     print("Unexpected error in bb()", sys.exc_info()[0])
        #     return -1
        return 1


    def execute(self):
        import PyNomad

        x0 = self.intial_beta
        #x0 = []
        lb = [item[0] for item in self.bound]
        ub = [item[1] for item in self.bound]
        # print(lb)
        # print(ub)
        # BB_MAX_BLOCK_SIZE parallel  'LH_SEARCH 0 0','SCALING ( 10 - - - 10 )'
        params = ['BB_OUTPUT_TYPE OBJ PB EB', 'MAX_BB_EVAL '+str(self.num_of_iter), 'BB_MAX_BLOCK_SIZE 2', 'DISPLAY_STATS BBE BLK_EVA OBJ',
                  'DIRECTION_TYPE ORTHO N+1','STATS_FILE nomad_output.txt BBE BLK_EVA OBJ']
        [x_return, f_return, h_return, nb_evals, nb_iters, stopflag] = PyNomad.optimize(self.input_bb, x0, lb, ub, params)
        print('\n NOMAD outputs \n X_sol={} \n F_sol={} \n H_sol={} \n NB_evals={} \n NB_iters={} \n'.format(x_return,
                                                                                                             f_return,
                                                                                                             h_return,
                                                                                                             nb_evals,
                                                                                                             nb_iters))
        print ('stop flag:', stopflag )




# -------------------------------------------- Main ---------------------------------------------------------
if __name__ == "__main__":
    # para_list = pd.read_excel('parameter.xls')
    tic = time.time()

    time_interval_demand = 15*60 # resolution of exit OD
    # define attributes:
    # True beta
    # beta = {'B_IN_VEHICLE_TIME':-0.147, 'B_NO_OF_TRANSFER':-0.573,'B_TRANSFER_OVER_DIST':-1.271, 'B_COMMONALITY_FACTOR':-3.679}
    # beta_bound_dict = {'in_vehicle_time':[-0.2, 0],'no_of_transfer':[-1, 0], 'transfer_over_dist':[-3, 0],
    #               'commonality_factor':[-5, 0]}
    beta_bound_dict = {'in_vehicle_time':[-2, 0],'no_of_transfer':[-4, 0], 'transfer_over_dist':[-6, 0],
                  'commonality_factor':[-10, 0]}
    intial_beta_dict = {'in_vehicle_time':-0.00001,'no_of_transfer':-0.0001, 'transfer_over_dist':-0.0001,
                  'commonality_factor':-0.0001}
    #---------------------------------------
    path_att = _DefaultValues.path_att  # names need to consistent with path attributes file
    #--------------------------
    beta_bound = [beta_bound_dict[key] for key in path_att]
    intial_beta_list = [intial_beta_dict[key] for key in path_att]
    num_of_parallel = 1
    num_of_iter = 102



    '''
        x1 * self.tb_path_attribute['in_vehicle_time'] +
        x2 * self.tb_path_attribute['no_of_transfer'] +
        x3 * self.tb_path_attribute['transfer_time'] +
        x4 * self.tb_path_attribute['commonality_factor'] +
    '''
    #===============
    date = '2017-03-16'
    name = '_syn'
    para_list = pd.read_csv('../0_user_configuration_parameter.csv').dropna()
    for ix, para in para_list.iterrows():
        TEST = 0
        Synthetic_data = 1 # 1 = test with synthetic data
        generate_demand = False # True if we only generate synthetic data not runing the model
        generate_beta = False
        time_interval_demand = 15*60 # resolution of exit OD
        sim_time_period = para['sim_time_period'].split('-')
        time_start = int(pd.to_timedelta(sim_time_period[0]).total_seconds())
        time_end = int(pd.to_timedelta(sim_time_period[1]).total_seconds())
        time_period_list = list(range(time_start, time_end, time_interval_demand))

        file_tail_name = date + name

        # ===============
        my_opt = OptimizationModel(para, time_interval_demand, path_att, TEST,file_tail_name, date, name)
        my_opt.Load_input_files(TEST = TEST, Synthetic_data = Synthetic_data, generate_data= generate_demand)
        my_opt.Load_input_files_SIM(TEST = TEST, Synthetic_data = Synthetic_data, generate_data= generate_beta)

        # ===============

        print('load file time:', time.time() - tic)

    #====Generate new demand synthetic====
    if generate_beta:
        print('----Generate file------')
        # True_beta_list = [-0.0663, -0.438, -0.183,-0.941]
        # Generate_synthetic_demand(my_opt, True_beta_list)
    else:
        print ('----Excute model------')
        #====Bayesian====
        # print('Bayesian_optmization')
        # # total = num_of_parallel*num_of_iter
        # random_seed = 4
        # output_file_name = 'Bayesian_results_EI_'+str(random_seed)+'.csv'
        # my_opt.out_put_name = output_file_name
        # out_put = BayesianOptSolver(num_of_parallel, num_of_iter, my_opt, beta_bound,output_file_name, random_seed).execute()
        # out_put = out_put.reset_index(drop=True)
        # out_put['Index'] = out_put.index
        # out_put.to_csv(output_file_name,index=False)
        #===========SPSA==========
        # print('SPSA_solver')
        # noise_var = 0.03
        # a_par = 0.01
        # random_seed = 1
        # output_name = 'SPSA_results_'+str(random_seed)+'.csv'
        # my_opt.out_put_name = output_name
        # if (num_of_parallel%2 == 1) and num_of_parallel!=1:
        #     print('num_of_parallel should be a even number for SPSA!')
        #     exit()
        # else:
        #     solver = SPSA_solver(intial_beta_list, num_of_parallel, num_of_iter,my_opt, beta_bound,noise_var,a_par, output_name, random_seed)
        #     solver.excute()
        # #===========NelderMead=============
        # print('NelderMead_optmization')
        # # The maximum parallel is 2*n, where n is the number of parameters
        # output_name = 'NelderMead_results_1.csv'
        # my_opt.out_put_name = output_name
        # solver = NelderMead(beta_ini = intial_beta_list, num_of_iter = num_of_iter, opt_model = my_opt, bound = beta_bound,output_name = output_name)
        # solver.execute()
        #======paulknysh_bb_solver===
        # print('CORS_RBF_solver')
        # random_seed = 2
        # output_name = 'CORS_RBF_results_' +str(random_seed) +'.csv'
        # fraction_global = 0.1
        # num_LT_search = math.ceil(fraction_global*num_of_iter)
        # print('number of global',num_LT_search)
        # num_follow_search = math.floor((1-fraction_global)*num_of_iter)
        # my_opt.out_put_name = output_name
        # CORS_RBF_solver(my_opt,beta_bound,random_seed,output_name,num_LT_search, num_follow_search,num_of_parallel) # define the parameters in function: paulknysh_bb_solver
        #====Test for beta====

        #beta_list = [[-0.0663, -0.438, -0.183,-0.941,-0.0767]] # Do not bother to use multiprocess
        ## beta_list = [[0, 0, 0, 0, 0]]  # Do not bother to use multiprocess
        ## beta_list = [[-0.1, -0.3334, -0.2138, -0.6390, -0.0839]]
        ## beta_list = [[-0.0798, -0.293, -0.210, -0.996, -0.0672]]
        # test_for_beta(my_opt, beta_list)

        #=======Nomad=====
        # print('Nomad_optmization')
        # # The maximum parallel is 2*n, where n is the number of parameters
        # num_of_parallel_nomad = 1
        # num_of_iter_nomad = 102
        # output_name = 'Nomad_output_6.csv'
        # my_opt.out_put_name = output_name
        # solver = Nomad_solver(num_of_parallel_nomad, num_of_iter_nomad, my_opt, beta_bound, intial_beta_list,output_name)
        # solver.execute()
        # ======
        print('Total_time:', round((time.time() - tic) / 60, 2), 'min')
        # print('Assignment finished!\n')
        # x = input('Press any key to exit ...')
