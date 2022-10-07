# define the default values in the assignment
# line_name = ['Airport Express Line', 'Disneyland Resort Line', 'East Rail Line', 'Island Line', 'Kwun Tong Line',
#              'Ma On Shan Line', 'South IsLand Line', 'Tseung Kwan O Line', 'Tsuen Wan Line', 'Tung Chung Line', 'West Rail Line']
# line_name_abbrev = {'Airport Express Line': 'AEL', 'Disneyland Resort Line': 'DRL', 'East Rail Line': 'EAL',
#                     'Island Line': 'ISL', 'Kwun Tong Line': 'KTL', 'Ma On Shan Line': 'MOL', 'South IsLand Line': 'SIL',
#                     'Tseung Kwan O Line': 'TKL', 'Tsuen Wan Line': 'TWL', 'Tung Chung Line': 'TCL', 'West Rail Line': 'WRL'}
# line_number = {'Airport Express Line': 10, 'Disneyland Resort Line': 16, 'East Rail Line': 17, 'Island Line': 13,
#                'Kwun Tong Line': 12, 'Ma On Shan Line': 18, 'South IsLand Line': 27, 'Tseung Kwan O Line': 15,
#                'Tsuen Wan Line': 11, 'Tung Chung Line': 14, 'West Rail Line': 19}

LINE_CODE_NAME = {16:'DRL',17:'EAL',13:'ISL', 12:'KTL', 14:'TCL', 18:'MOL', 15:'TKL', 11:'TWL', 19:'WRL', 27:'SIL',10: 'AEL'}
LINE_CODE_LIST = [16, 17, 13, 12, 14, 18, 15, 11, 19, 27,10] # for process data
REFERENCE_TIME = '00:00:00'   # Referee time is 00:00:01 am
CHOICE_INTERVAL = 30 * 60     # Route choice time interval is in 30 minutes
DEFAULT_HEADWAY = 4 * 60      # in seconds

DEFAULT_COOL_DOWN_TIME = 30   # in minute
DEFAULT_WARM_UP_TIME = 30     # in minute

# Default global values
RND_SEED = 1
TIME_INTERVAL_CHOICE = 30 * 60                  # choice fractions calculated every 30 minutes
TIME_INTERVAL_DEMAND = 15 * 60                  # Demand time interval for optimization objective calculation
DEFAULT_ACCESS_TIME = 60                        # Default access time 30 seconds
DEFAULT_EGRESS_TIME = 60                        # Default egress time 30 seconds
DEFAULT_TRANSFER_TIME = 30                      # Default transfer time 30 seconds
#-------------For TEST-------------------
#DEFAULT_CAR_CAP_1 = 10                          # Default capacity1 230 pax per car for normal station
#DEFAULT_CAR_CAP_2 = 10                     # Default capacity2 270 pax per car for congested station
#----------------------------------
DEFAULT_CAR_CAP_1 = 230                           # Default capacity1 230 pax per car for normal station
DEFAULT_CAR_CAP_2 = 248                        # Default capacity2 270 pax per car for congested station

EFFECTIVE_CAP_PARA_LOAD = 0.0904456797440875                  # cap = DEFAULT_CAR_CAP_1 + EFFECTIVE_CAP_PARA_LOAD*train load + EFFECTIVE_CAP_PARA_WAIT * waiting pax
EFFECTIVE_CAP_PARA_WAIT = 0.0718080913445427
#------------------------------
DEFAULT_CAP_2_STATION = [(2,11,1),(3,11,1),(4,11,1)]         # Congested station list: (station, line, direction)
# DEFAULT_CAP_2_STATION = [(0,1,1)]         # Congested station list: (station, line, direction)
MAXIMUM_PATH_NUM = 10                           # Default maximum number of paths per od pair
RND_FLAG = 1 # Important! determine wheter the model is random (if =0: access time = 0  egress time = 0 transfer time = transfer times)
DEMAND_SCALE = 1 # default = 1, use 10 the avoid the small demand resulting in difference between TEN and SIM, should be integer
# Discrete choice model settings
#  in_vehicle_time, no_of_transfer, waiting time, transfer_time, commonality_factor
# Weixuan's model: -0.18439, -0.4844


path_att = ['in_vehicle_time', 'no_of_transfer', 'transfer_over_dist',
            'commonality_factor']

