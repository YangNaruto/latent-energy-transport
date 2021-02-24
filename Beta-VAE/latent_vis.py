import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
# plt.rcParams.update({'font.size': 16})
sns.set()
# sns.set_style('white')

male_sequences = np.load('male_gen.npy')
female_sequences = np.load('female_gen.npy')

steps, num_sample, dim = male_sequences.shape

male_change = np.zeros((num_sample, dim))
female_change = np.zeros((num_sample, dim))

for s in range(steps-1):
	male_change += np.abs(male_sequences[s+1] - male_sequences[s])

for s in range(steps-1):
	female_change += np.abs(female_sequences[s+1] - female_sequences[s])

# (figure, ax) = plt.subplots(1, 2, figsize=(12, 1))
# # plt.tight_layout()
# ax[0] = sns.heatmap(np.mean(male_change, axis=0, keepdims=True), ax=ax[0], cbar=False)
# ax[0].yaxis.set_major_locator(plt.NullLocator())
# ax[0].xaxis.set_major_locator(plt.NullLocator())
# # ax[0].set_ylabel('1000 Samples', fontdict={'size':16})
# # ax[0].set_xlabel('$z$', fontdict={'size':16})
# # ax[0].set_title('male to female')
#
# ax[1] = sns.heatmap(np.mean(female_change, axis=0, keepdims=True), ax=ax[1], cbar=True)
# plt.gca().yaxis.set_major_locator(plt.NullLocator())
# plt.gca().xaxis.set_major_locator(plt.NullLocator())
# # ax[1].set_xlabel('$z$', fontdict={'size':16})
# # ax[1].set_title('female to male')
# plt.savefig('cbar.png', dpi=300)
# plt.show()

(figure, ax) = plt.subplots(1, 2, figsize=(14, 6))
x = np.arange(1, 33)
male_mean = np.mean(male_change, axis=0)
male_std = np.std(male_change, axis=0)
ax[0].plot(x, male_mean, color=(0/255, 112/255, 192/255))
ax[0].fill_between(x, male_mean - male_std, male_mean + male_std,
                 color=(0/255, 112/255, 192/255), alpha=0.2)
ax[0].set_ylim(0, 0.6)
ax[0].set_xlim(0, 33)

ax[0].set_xlabel('$z$', fontdict={'size':16})
ax[0].set_ylabel('Absolute Shift', fontdict={'size':16})
female_mean = np.mean(female_change, axis=0)
female_std = np.std(female_change, axis=0)
ax[1].plot(x, female_mean, color=(192/255, 0/255, 0/255))
ax[1].fill_between(x, female_mean - female_std, female_mean + female_std,
                 color=(192/255, 0/255, 0/255), alpha=0.2)
ax[1].set_ylim(0, 0.6)
ax[1].set_xlim(0, 33)
ax[1].set_xlabel('$z$', fontdict={'size':16})
# ax[1].set_ylabel('Absolute Shift', fontdict={'size':16})
plt.savefig('betavae-var.png', dpi=300)

plt.show()
