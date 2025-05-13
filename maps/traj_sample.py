import pandas as pd
import numpy as np
import os
import sys

traj_dir = sys.argv[1]
sub_dirs = os.listdir(traj_dir)
indices = np.linspace(0, 4, )
for sub in sub_dirs:
    sub_dir = os.path.join(traj_dir, sub)
    names = os.listdir(sub_dir)
    names.sort()
    for i in range(len(names)):
        name = names[i]
        print(name)
        csv_file = os.path.join(sub_dir, name)
        df = pd.read_csv(csv_file, encoding='utf-8')
        indices = np.array([i*5 for i in range(len(df)//5)])
        sampled_df = df.iloc[indices].reset_index(drop=True)
        sampled_df.to_csv(os.path.join(sub_dir, sub+'_'+str(i)+'.csv'))
