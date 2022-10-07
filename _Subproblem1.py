"""
This is the module of solve subproblem 1 (large scale LP)

Baichuan Mo
"""
import numpy as np
import time
from gurobipy import *
import itertools


class OPT_Path_share(object):

    def __init__(self, entry_demand, path_set, NODE, cm, jn_list, q_true, p_constraints, p_refered):
        """
        :param
        Entry_demand: Entry OD dictionary, keys: (TEN_O, D), values: number of trips
        :return: Optimal path fraction given
        """
        self.entry_demand = entry_demand
        self.path_set = path_set
        self.cm = cm
        self.jn_list = jn_list
        self.NODE = NODE
        self.q_ture = q_true
        self.p_constraints = p_constraints
        self.p_refered = p_refered
        # self.bound = bound
    def main_model(self):
        q = {} # variable TEN OD flow
        p = {} # variable path share (i,j,r) #simple case first, no dynamic here
        p_refer_dict = {}
        m = Model("SubProb1")
        # m.setParam('OutputFlag', False) # false = not out put
        OD_pair = []
        TEN_od_pair = []
        tic = time.time()
        print ('start add q constraints')
        for key in self.entry_demand:
            origin_station = self.NODE.loc[key[0],'station']
            destination_station = key[1]
            im = key[0]
            if (origin_station, destination_station) not in self.path_set:  # no record of this OD pair (bug in process branch lines, fixed in the future)
                continue
            r_list={}
            for r in self.path_set[(origin_station, destination_station)]:
                if (origin_station, destination_station, r) not in p:
                    p[(origin_station, destination_station, r)] = m.addVar(vtype=GRB.CONTINUOUS, name='p_'+str(origin_station)+'_'
                    + str(destination_station) + '_'+str(r), lb=0.0, ub=1.0)
                    p_refer_dict[(origin_station, destination_station, r)] = self.p_refered.loc[(self.p_refered['origin']==origin_station)&
                                                                                                (self.p_refered['destination'] == destination_station)&
                                                                                                (self.p_refered['path_id'] == r),'path_share'].values[0]
                if (im, destination_station, r) not in self.jn_list: # no record of this entry OD for this path
                    continue
                for jn in self.jn_list[(im, destination_station, r)]:
                    if (im, jn) not in TEN_od_pair:
                        TEN_od_pair.append((im, jn))
                        q[(im, jn)] = m.addVar(vtype=GRB.CONTINUOUS, name='q_'+str(im)+'_'+str(jn), lb=0.0)
                    if jn not in r_list:
                        r_list[jn]=[r]
                    else:
                        r_list[jn].append(r)
                    #
                    # if (im,jn,r) not in self.cm: # no record
                    #     self.cm[(im,jn,r)] = 0
            for jn in r_list:
                m.addConstr(q[(im, jn)] == quicksum([self.entry_demand[key]*p[(origin_station, destination_station, r)]*self.cm[(im,jn,r)] \
                                                 for r in r_list[jn]]))
            if (origin_station,destination_station) not in OD_pair:
                OD_pair.append((origin_station,destination_station))
                m.addConstr(quicksum(p[(origin_station, destination_station, r)] for r in self.path_set[(origin_station, destination_station)]) == 1)
        print('finish add q constraints, time', time.time() - tic, 'sec')
        # Add p constraints, linear approximation of logit model:
        # print('start add p constraints')
        tic2 = time.time()

        for key in self.p_constraints:
            # find first available as base, all odr in this group have equal path share
            current_id = -1
            for i in range(0, len(self.p_constraints[key])):
                base = self.p_constraints[key][i]
                if (base[0], base[1], base[2]) in p:
                    current_id = i
                    break
            if current_id!=-1: #successfully find a base
                for i in range(current_id + 1,len(self.p_constraints[key])):
                    odr = self.p_constraints[key][i]
                    if (odr[0], odr[1], odr[2]) in p:
                        m.addConstr(p[(odr[0], odr[1], odr[2])] == p[(base[0], base[1], base[2])])
        print('finish add p constraints, time', time.time() - tic2, 'sec')
        #------Objective function----
        for im_jn in TEN_od_pair:
            if (im_jn[0],im_jn[1]) not in self.q_ture:
                self.q_ture[(im_jn[0],im_jn[1])] = 0
        w1 = 1
        w2 = 0
        m.setObjective(w1*quicksum([(q[(im_jn[0],im_jn[1])]-self.q_ture[(im_jn[0],im_jn[1])])*(q[(im_jn[0],im_jn[1])]-self.q_ture[(im_jn[0],im_jn[1])])\
                                 for im_jn in TEN_od_pair]) + w2*quicksum( (p_refer_dict[odr] - p[odr])*(p_refer_dict[odr] - p[odr]) for odr in p_refer_dict), GRB.MINIMIZE)
        print ('input model time:', time.time() - tic)
        m.optimize()
        # if m.status == GRB.OPTIMAL:
        p_new = {}
        q_est = {}
        for pair, var in p.items():
            p_new[pair] = var.X
        for pair, var in q.items():
            q_est[pair] = var.X
        return p_new, q_est, m.objVal
        # else:
        #     m.computeIIS()
        #     m.write("infeasible.ilp")
        #     print('Model is infeasible, pls check infeasible.ilp')
        # print (self.q_ture)
        # print (q)
