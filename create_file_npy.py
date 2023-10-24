#! D:\units\thesis\thesis\train_reward_inverse_rl_w_sys_model\.venv\Scripts\python.exe

import numpy as np
import os
import pandas as pd
 

df2 = []
import os

rootdir ='D:\\units\\thesis\\thesis\\train_reward_inverse_rl_w_sys_model'

for subdir, dirs, files in os.walk(rootdir):
    if 'Run_' in subdir:
        
        df2_row = []
        states, traj_probs, actions = [], [], []
        filepath1 = subdir + os.sep +  'actions.csv'
        filepath2 = subdir + os.sep + 'states.csv'
        
        actions_df = pd.read_csv(filepath1, header = None).to_numpy()
        states_df = pd.read_csv(filepath2, header = None).to_numpy()
        for a in actions_df:
            actions.append(a[0])

        for s in states_df:
            states.append(s)

        traj_probs = np.ones((500,4))

        # Actions were mapped to 1,2,3,4 to facilitate policy network config, but has to be re-mapped to original values, 
        # The system model takes as input vectors
        my_dict = {0:[0,2400,0], 1:[0,2400,1000], 2:[2300,0,0], 3:[2300,0,1000]}
        actions_arr = np.zeros((500,3))  
        actions_arr = map(my_dict.get, actions)
 
        df2_row = [states, np.array(list(actions_arr)), traj_probs]
        df2.append(df2_row)


df2 = np.array(df2,dtype=object)

np.save('dataset', df2)
