import numpy as np
rng = np.random.default_rng()
arr1 = np.arange(9).reshape(3, 3)
arr2 = np.arange(9).reshape(3, 3)
print(arr1)
print(arr2)
rng.shuffle(arr1, axis=1)
rng.shuffle(arr2, axis=0)
print(arr1)
print(arr2)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

npz = np.load('./assets/dataset/X100/RasSignImg_X100.npz')
df= pd.DataFrame.from_dict({item: npz[item] for item in npz.files}, orient='index')
# monthly_df = pd.DataFrame(df.groupby(['month', 'period'])['sales'].sum())
pivot_monthly_df = df.reset_index().pivot(index='Y', columns='period', values='sales')
pivot_monthly_df
