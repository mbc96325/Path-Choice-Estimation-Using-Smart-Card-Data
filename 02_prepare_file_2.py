'''
Version 4: use merge to improve searching path speed
'''
import _DefaultValues
import pandas as pd
import time
import os
import functools
import math
import multiprocessing as mp
import numpy as np
import warnings
from itertools import chain
from _postprocess_mtr_network_operation_ver2 import post_process
warnings.simplefilter("ignore")

def fast_concate(frames):
    def fast_flatten(input_list):
        return list(chain.from_iterable(input_list))
    COLUMN_NAMES = frames[0].columns
    df_dict = dict.fromkeys(COLUMN_NAMES, [])
    for col in COLUMN_NAMES:
        # Use a generator to save memory
        extracted = (frame[col] for frame in frames)

        # Flatten and save to df_dict
        df_dict[col] = fast_flatten(extracted)
    df = pd.DataFrame.from_dict(df_dict)[COLUMN_NAMES]
    return df

def process_afc_sjsc(df, flag):
    # 1. Data formatting
    # Formatting the columns names, change the timestamp format to YYYY-MM-DD,
    # and extract useful columns for further use
    # Output formats: user_id, txn_dt, txn_type_co,txn_subtype_co,entry_stn,txn_loc,txn_audit_no
    # Input formats:
    #     octopus: csc_phy_id,business_dt,txn_dt,txn_type_co,txn_subtype_co,train_entry_stn,txn_loc,txn_audit_no,
    #              hw_type_co,mach_no,train_direct_ind,txn_value,modal_disc_value,csc_rv_value
    #     sjtc: sjsc_id,business_dt,txn_dt,txn_type_co,txn_subtype_co,entry_stn,txn_loc,txn_seq_no,
    #           mach_no,recycle_count,prom_code,txn_value
    df.columns = map(str.lower, df.columns)

    df['txn_dt'] = df['txn_dt'].str.split(' ')
    df['txn_dt'] = df['txn_dt'].apply(lambda x: x[1])
    if flag == 'octopus':
        df.rename(columns={'csc_phy_id': 'user_id', 'train_entry_stn': 'entry_stn'}, inplace=True)
    elif flag == 'sjtc':
        df.rename(columns={'sjsc_id': 'user_id', 'txn_seq_no': 'txn_audit_no'}, inplace=True)

    # 2. construct
    return df[['user_id', 'txn_dt', 'txn_type_co', 'txn_subtype_co', 'entry_stn', 'txn_loc', 'txn_audit_no']]

class External_file(object):

    def __init__(self, file_path, file_path_output, file_time_table, file_network_operation, od_file, empty_arrangement, transfer_time_file, sim_time_period, num_of_core):
        # initialize the variables
        # parallel
        self.num_of_core = num_of_core
        #
        
        self.time_period = [sim_time_period[0] - _DefaultValues.DEFAULT_WARM_UP_TIME * 60,
                            sim_time_period[1] + _DefaultValues.DEFAULT_COOL_DOWN_TIME * 60] # warm up 60 min; cool down 60 min
        self.file_path_output = file_path_output
        #
        self.default_transfer_walk_time = _DefaultValues.DEFAULT_TRANSFER_TIME

        #--
        self.time_table = pd.read_csv(file_path + file_time_table, low_memory=False).fillna(0)
        # get end station of each trip
        temp = self.time_table.groupby(['LINE_CODE','Trip_No']).apply(lambda x: list(zip(list(x['From_ID']),list(x['To_ID'])))).reset_index().rename(columns={0:'train_passing_link_list'})
        self.time_table = self.time_table.merge(temp, left_on = ['LINE_CODE','Trip_No'], right_on = ['LINE_CODE','Trip_No'], how='inner')
        # convert time
        self.time_table['Arr_From_ID'] = pd.to_timedelta(self.time_table['Arr_From']).dt.total_seconds().apply(int)
        self.time_table['Dep_From_ID'] = pd.to_timedelta(self.time_table['Dep_From']).dt.total_seconds().apply(int)
        self.time_table['Arr_To_ID'] = pd.to_timedelta(self.time_table['Arr_To']).dt.total_seconds().apply(int)
        self.time_table['Dep_To_ID'] = pd.to_timedelta(self.time_table['Dep_To']).dt.total_seconds().apply(int) 
            
        # select used time table
        self.time_table = self.time_table.loc[(self.time_table['Dep_To_ID']>=self.time_period[0])&
                                              (self.time_table['Arr_From_ID']<=self.time_period[1])].reset_index(drop=True)

        self.link_all_list = list(set(zip(list(self.time_table['From_ID']), list(self.time_table['To_ID']))))
        self.link_table = pd.DataFrame()
        self.link_temp = pd.DataFrame()
        
        self.network_operation = file_network_operation
        try:
            self.tb_txn = pd.read_csv(self.file_path_output + 'tb_txn.csv')
            self.tb_itinerary = pd.read_csv(self.file_path_output + 'tb_itinerary.csv')
        except:
            print('tb_txn or tb_itinerary not find, run 01_1_prepare file first')
            exit()

        #---


        # reading and processing the external files
        try:
            self.empty_arrangement = pd.read_csv(file_path + empty_arrangement)
            self.empty_arrangement = self.empty_arrangement.loc[self.empty_arrangement['date']==DATE]
            self.empty_arrangement['time_stamp'] = pd.to_timedelta(self.empty_arrangement['time']).dt.total_seconds().apply(int) 
        except:
            self.empty_arrangement = []   
        
        try:
            self.transfer_time_file = pd.read_csv(file_path + transfer_time_file)
            # generate transfer time table: platform to platform
            self.transfer_time_file = self.create_transfer_time_table()
        except:
            self.transfer_time_file = pd.DataFrame(columns=['platform_x', 'platform_y', 'walking_time'])
            
        #print (self.time_table)
        #print (self.file_od_matrix)
        #-----
    def create_transfer_time_table(self):
        # now do nothing. wait for more data
        return self.transfer_time_file
        
    def Share_period(self,time):
        if time >5*3600 + 15*60 and time <= 10*3600:  #"5:15"-"10:00"
            val_col = 'MP_SHARE'
        elif time >10*3600 and time <= 16*3600: #start == "10:15" and end == "16:00":
            val_col = 'OP_SHARE'
        elif time >16*3600 and time <= 19*3600 + 30*60: #start == "16:15" and end == "19:30":
            val_col = 'EP_SHARE'
        elif time >1*3600 and time <= 5*3600: #start == "1:15" and end == "5:00":
            val_col = 'OP_SHARE'
        elif time >19*3600 + 30*60 and time <= 23*3600: #start == "19:45" and end == "23:00":
            val_col = 'OP_SHARE'
        else:
            val_col = 'OTHER_SHARE'
        return val_col
    
    def generate_input_files(self):
        print('generating TEN_Node and TEN_Link ...')
        tic = time.time()
        Node_table_temp = self.create_TEN_Node()
        Link_table_temp = self.create_TEN_Link()
        self.link_temp = Link_table_temp
        self.node_table, self.link_table, self.node_interval = self.create_TEN_Node_and_Link(Link_table_temp, Node_table_temp)


        self.node_interval.to_csv(self.file_path_output + 'tb_node_interval.csv', index=False)

        tic = time.time()
        self.create_TEN_demand()
        print('Finish Create TEN OD Demand with time:', time.time() - tic)
        #----------------demand interval----------------
        tic = time.time()
        self.create_demand_interval()
        print('Finish Create OD Demand Interval with time:', time.time() - tic)


    def create_TEN_Node_and_Link(self,Link_table_temp,Node_table_temp):

        #-----Derive Link table-----
        Link_table_temp = Link_table_temp.merge(Node_table_temp, left_on='start_node_property', right_on='Property', how='inner')[['start_node_property', 'end_node_property','type','Car_Num','Link_index','Node_index','time','station']]
        Link_table_temp = Link_table_temp.rename(columns={'Node_index': 'start_node','time': 'start_time', 'station': 'start_station'})
        Link_table_temp = Link_table_temp.merge(Node_table_temp, left_on='end_node_property', right_on='Property', how='inner')[['start_node_property', 'end_node_property','type','Car_Num','Link_index','start_node','start_time','Node_index','time','start_station','station', 'line','direction']]
        Link_table = Link_table_temp.rename(columns={'Node_index': 'end_node','time': 'end_time','station': 'end_station'})[['Link_index','Car_Num','type','start_node','end_node','start_time','end_time','start_station','end_station', 'line','direction']]
        #print (Link_table)
        #-----Derive node table-----

        Node_table = Node_table_temp[['Node_index','station', 'line','direction','time']]
        Node_interval = Node_table.copy()
        Node_interval['time_interval'] = Node_interval['time'] // _DefaultValues.TIME_INTERVAL_DEMAND * _DefaultValues.TIME_INTERVAL_DEMAND
        Node_interval = Node_interval.drop(columns = ['Node_index','time','line','direction']).drop_duplicates()
        Node_interval = Node_interval.sort_values(by=['station','time_interval'])
        Node_interval = Node_interval.reset_index(drop=True)
        Node_interval['Node_index'] = Node_interval.index
        #print (Node_table)
        return Node_table, Link_table, Node_interval

    def create_TEN_Node(self):
        '''
        Time-space Extended Network: Node table; 
        columns=['station', 'line','direction','time']
        '''
        From_node_table = pd.DataFrame(index=range(0, len(self.time_table)),
                                   columns=['station', 'line','direction','time','Property'])            
        From_node_table['station'] = self.time_table['From_ID']
        From_node_table['Trip_No'] = self.time_table['Trip_No']
        From_node_table['line'] = self.time_table['LINE_CODE']
        From_node_table['time'] = self.time_table['Dep_From_ID']
        From_node_table['direction'] = self.time_table['Direction_ID']
        #print (From_node_table)
        From_node_table['Property'] = From_node_table['station'].apply(str) + '_' + From_node_table['line'].apply(str) + '_' \
        + From_node_table['time'].apply(str) + '_' + From_node_table['direction'].apply(str)
        
        
        To_node_table = pd.DataFrame(index=range(0, len(self.time_table)),
                                   columns=['station', 'line','direction','time','Property'])            
        To_node_table['station'] = self.time_table['To_ID']
        To_node_table['Trip_No'] = self.time_table['Trip_No']
        To_node_table['line'] = self.time_table['LINE_CODE']
        To_node_table['time'] = self.time_table['Dep_To_ID']
        To_node_table['direction'] = self.time_table['Direction_ID']
        To_node_table['Property'] = To_node_table['station'].apply(str) + '_' + To_node_table['line'].apply(str) + '_' \
        + To_node_table['time'].apply(str) + '_' + To_node_table['direction'].apply(str)
        
        Node_table = pd.concat([From_node_table, To_node_table]).drop_duplicates(keep='last')    
        Node_table = Node_table.sort_values(by=['time'])
        Node_table = Node_table.groupby(['line','direction','Trip_No']).apply(lambda x: x.sort_values(["time"], ascending=True)).reset_index(drop=True)
        # Node_table = Node_table.reset_index(drop=True)
        Node_table['Node_index'] = Node_table.index

        #print (Node_table)
        #print (len(Node_table))        
        # save to file
        
        return Node_table

    def create_TEN_Link(self): 
        Link_table = pd.DataFrame(index=range(0, len(self.time_table)),
                                   columns=['start_node_property', 'end_node_property','type','line','direction','link_start_time','link_end_time',
                                            'link_start','link_end','Trip_No'])



        Link_table['type'] = 1 # 1: in-veh  2: Transfer 3:left behind
        Link_table['line'] = self.time_table['LINE_CODE'].apply(int)
        Link_table['direction'] = self.time_table['Direction_ID'].apply(int)
        Link_table['link_start_time'] = self.time_table['Dep_From_ID']
        Link_table['link_end_time'] = self.time_table['Dep_To_ID'].apply(int)
        Link_table['link_start'] = self.time_table['From_ID'].apply(int)
        Link_table['link_end'] = self.time_table['To_ID'].apply(int)

        # cap2_station = [item[0] for item in _DefaultValues.DEFAULT_CAP_2_STATION]
        # cap2_line = [item[1] for item in _DefaultValues.DEFAULT_CAP_2_STATION]
        # cap2_dir = [item[2] for item in _DefaultValues.DEFAULT_CAP_2_STATION]

        Link_table['Car_Num'] = self.time_table['Car_Num']

        Link_table['Trip_No'] = self.time_table['Trip_No']
        Link_table['train_passing_link_list'] = self.time_table['train_passing_link_list']
        Link_table['start_node_property'] = self.time_table['From_ID'].apply(str) + '_' + self.time_table['LINE_CODE'].apply(str) + '_' \
        + pd.to_timedelta(self.time_table['Dep_From']).dt.total_seconds().apply(int).apply(str) + '_' + self.time_table['Direction_ID'].apply(str)
        Link_table['end_node_property'] = self.time_table['To_ID'].apply(str) + '_' + self.time_table['LINE_CODE'].apply(str) + '_' \
        + pd.to_timedelta(self.time_table['Dep_To']).dt.total_seconds().apply(int).apply(str) + '_' + self.time_table['Direction_ID'].apply(str) 

        Link_table['link_property'] = Link_table['direction'].apply(str) + '_' + Link_table['line'].apply(str) + '_' \
        + Link_table['link_start'].apply(str) + '_' + Link_table['link_start_time'].apply(str) + '_' + Link_table['link_end'].apply(str) + '_' + Link_table['link_end_time'].apply(str)

        Link_table = Link_table.groupby(['line', 'direction', 'Trip_No']).apply(
            lambda x: x.sort_values(["link_start_time"], ascending=True)).reset_index(drop=True)
        Link_table['Link_index'] = Link_table.index
        if len(self.empty_arrangement) > 0: # make empty train with 0 capacity
            for ind, item in self.empty_arrangement.iterrows():
                temp = Link_table.loc[(Link_table.line == item.line) &
                                  (Link_table.direction == item.direction) &
                                  (Link_table.link_start == item.station)]

                temp['time_diff'] = abs(temp.link_start_time - item.time_stamp)
                index_empty = temp['time_diff'].idxmin()
                if temp.loc[index_empty, 'time_diff'] < 5*60:   # within 5 minutes
                    Link_table.loc[index_empty,'Car_Num'] = 0  #

        return Link_table


    def create_TEN_demand(self):
        # -----begin to generate tb_demand-----
        tic = time.time()
        # station_departure_list = self.time_table.loc[:, ['From_ID', 'Dep_From_ID', 'LINE_CODE',
        #                                       'Direction_ID']].drop_duplicates().reset_index(drop=True).\
        #     rename(columns = {'Dep_From_ID':'event_time','From_ID':'origin_station','LINE_CODE':'first_line','Direction_ID':'first_direction'})
        #
        # # match with itenerary
        #-----
        station_departure = self.tb_itinerary.loc[self.tb_itinerary.itinerary==1]
        check_error = station_departure.loc[station_departure['origin'] != station_departure['boarding_station']]
        if len(check_error)>0:
            print('origin != boarding station exist, fix it first')
            exit()


        station_departure = station_departure.merge(self.time_table, left_on = ['origin','line','direction'],
                                                                    right_on = ['From_ID','LINE_CODE','Direction_ID'],
                                                    how = 'left')
        check_error = station_departure.loc[(station_departure['Arr_From_ID'].isna())&
        (station_departure['origin']!=70)]
        if len(check_error)>0:
            print('Exist not reasonable boarding station in itinerary')
            print('This boarding station cannot be found in time table')
            exit()
        station_departure = station_departure.rename(columns = {'origin':'origin_station',
                                                                'Dep_From_ID':'event_time',
                                                                'LINE_CODE':'first_line',
                                                                'Direction_ID':'first_direction'})
        station_departure_list = station_departure.loc[:,['origin_station','event_time','first_line','first_direction']].drop_duplicates()
        # station_departure_list = pd.merge(station_departure_list, self.tb_itinerary,
        #                          left_on=['origin_station','first_line','first_direction'],
        #                          right_on=['origin', 'line','direction'], how='right')
        # temp = station_departure_list.loc[station_departure_list['first_line'].isna(),
        #        :]  # fixed 1 39 problem, assign a event list to these
        #--------------------------
        station_departure_list['event_station'] = station_departure_list['origin_station'].apply(str) + '_' + \
                                                  station_departure_list['first_line'].apply(str) + '_' + \
                                                  station_departure_list['first_direction'].apply(str)

        tb_txn = self.tb_txn
        tb_txn['flow'] = 1
        def get_event_upper_lower(event_list):
            event_list = event_list.sort_values(by=['event_station', 'event_time']).rename(
                columns={'event_time': 'event_time_lower'}).reset_index(drop=True)
            event_list['index'] = event_list.index
            event_list_new = event_list.loc[:, ['index', 'event_station', 'event_time_lower']].rename(
                columns={'event_time_lower': 'event_time_upper'})
            event_list_new['index'] -= 1
            event_list = event_list.merge(event_list_new, left_on=['index', 'event_station'],
                                          right_on=['index', 'event_station'])
            return event_list
        station_departure_list = get_event_upper_lower(station_departure_list)

        OD_grouped = tb_txn.groupby(['pax_origin'])
        demand = []
        for idx, info in OD_grouped:
            event_list_temp = station_departure_list.loc[station_departure_list['origin_station'] == idx]
            temp = event_list_temp.merge(
                info[['pax_origin', 'pax_destination', 'flow', 'pax_tapin_time']],
                left_on=['origin_station'], right_on=['pax_origin'])
            temp = temp.loc[
                (temp['pax_tapin_time'] >= temp['event_time_lower']) & (
                            temp['pax_tapin_time'] < temp['event_time_upper'])]
            temp = temp.groupby(
                ['pax_origin', 'pax_destination', 'first_line','first_direction','event_time_lower', 'event_time_upper']).\
                sum()[['flow']].reset_index()
            demand.append(temp)

        demand = fast_concate(demand)
        demand = demand.sort_values(by=['pax_origin','pax_destination','first_line','first_direction','event_time_lower','event_time_upper'])
        demand = demand.rename(columns = {'pax_origin':'origin_station','pax_destination':'destination_station','event_time_lower':'first_time'})
        self.tb_demand = demand.loc[demand['origin_station']!=demand['destination_station']]
        self.tb_demand['flow'] *= _DefaultValues.DEMAND_SCALE # scale the demand
        # temp = self.tb_demand.loc[self.tb_demand['first_line'].isna(),:] # fixed 1 39 problem, assign a event list to these
        print('generate demand time:', time.time() - tic)
        self.tb_demand.to_csv(self.file_path_output + 'tb_demand.csv', index=False)

    def create_demand_interval(self):
        # -----begin to generate tb_demand-----

        tb_txn = self.tb_txn
        tb_txn['flow'] = 1
        tb_txn['entry_interval'] = tb_txn['pax_tapin_time'] // _DefaultValues.TIME_INTERVAL_DEMAND * _DefaultValues.TIME_INTERVAL_DEMAND
        tb_txn = tb_txn.rename(columns = {'pax_origin':'origin','pax_destination':'destination'})
        tb_demand_interval = tb_txn.groupby(['origin','destination','entry_interval']).sum()[['flow']].reset_index(drop=False)
        tb_demand_interval.to_csv(self.file_path_output + 'tb_demand_interval.csv', index=False)




def main(para, TEST, date, name):
    #print (para)
    global CAPACITY_CAR
    global TRANSFER_WALKING_TIME
    global DATE
    

    DATE = date

    file_path_external = 'External_data/'

    #--real data----
    if TEST == 0:
        sim_time_period = para['sim_time_period'].split('-')
        time_start = int(pd.to_timedelta(sim_time_period[0]).total_seconds())
        time_end = int(pd.to_timedelta(sim_time_period[1]).total_seconds())
        sim_time_period = [time_start, time_end]
        TRANSFER_WALKING_TIME = _DefaultValues.DEFAULT_TRANSFER_TIME
        file_time_table = 'Timetable_Weekday_2017-03-16.csv'
        mtr_raw_assignment_file_path = file_path_external + 'mtr_network_operation_assignment.csv'
        user_configuration = '0_user_configuration_parameter.csv'
        network_file = post_process(user_configuration, mtr_raw_assignment_file_path,output=True)
        file_network_operation = network_file
        afc_file = 'AFC_TXN_' + date + '.csv'
        sjsc_file = 'SJSC_TXN_'+ date + '.csv'
        od_file = [afc_file, sjsc_file]
        transfer_time_file = 'Transfer_Walking_Time.csv'
        if para['operation_arrangement'] == 'Empty':
            empty_arrangement = 'Empty_Train_Arrangement.csv'
        else:
            empty_arrangement = ''
        num_of_core = 22
        file_path_output = 'Assignment_' + str(time_start) + '-' + str(time_end) + '_' + date + name + '/'

    #--test data----
    else:
        sim_time_period = [int(pd.to_timedelta('7:00:00').total_seconds()),int(pd.to_timedelta('8:00:00').total_seconds())]
        TRANSFER_WALKING_TIME = 5*60
        file_time_table = 'Time_table_TEST.csv'
        mtr_raw_assignment_file_path = file_path_external + 'MRT_Network_Operation_TEST.csv'
        user_configuration = '0_user_configuration_parameter.csv'
        file_network_operation = post_process(user_configuration, mtr_raw_assignment_file_path, output=True)
        num_of_core = 2
        CAPACITY_CAR = 2
        empty_arrangement = ''
        afc_file = 'AFC_TXN_TEST.csv'
        sjsc_file = 'SJSC_TXN_TEST.csv'
        od_file = [afc_file, sjsc_file]
        transfer_time_file = ''
        file_path_output = \
            'Assignment_test/'

    #--------
    folder = os.path.exists(file_path_output)
    if not folder:                   
        os.makedirs(file_path_output)        
    external_file = External_file(file_path_external,file_path_output, file_time_table, file_network_operation, od_file, empty_arrangement, transfer_time_file, sim_time_period, num_of_core)
    
    external_file.generate_input_files()

#================================


if __name__ == "__main__":
    #os.chdir('F:\\01_Network Performance Model\Code') #change current file path
    para_list = pd.read_csv('0_user_configuration_parameter.csv').dropna()
    tic=time.time()
    TEST = 0 #1 means run with test data
    date = '2017-03-16'
    name = '_syn'
    for ix, para in para_list.iterrows():
        main(para, TEST, date, name)
    print('Data Processing time:', time.time() - tic, 'sec')
