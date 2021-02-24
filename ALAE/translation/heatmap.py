import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')

file_dir = 'C:\\Users\\Yang\\Desktop\\celeba-hq1024\\representatives\\male\\008000'
files = list(glob(f'{file_dir}/*.npy'))
stacked_data = np.zeros((len(files), 512))
for i, file in enumerate(files):
	array = np.squeeze(np.load(file), axis=1)
	pointer = array[0]
	change = np.zeros((1, 512))
	for j in range(15):
		change += np.abs(array[j] - pointer)
		pointer = array[j]
	stacked_data[i:i+1] = change / 15

plt.figure(figsize=(5, 5))

ax = sns.heatmap(stacked_data)
# ax.yaxis.set_major_locator(ticker.MultipleLocator(200))
# ax.yaxis.set_ticks_position('none')
# plt.gca().yaxis.set_major_locator(plt.NullLocator())
# plt.ylabel('Steps', fontdict={'size':16})
# plt.xlabel('$z$', fontdict={'size':16})
plt.xticks(fontsize= 14 )
plt.savefig(f'heat.png', dpi=300)
plt.close()
