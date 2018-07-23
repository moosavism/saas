import numpy as np
from matplotlib import pyplot as plt
plt.close("all")

# plot losses_all
colors = ['blue','green','red','cyan','magenta']

file_name = "training_losses.npy"
logs = np.save('file_name', logs)

font = 15
linew = 3

fig = plt.figure()
ax = plt.subplot(111)
legends = []
last = 30
for i, loss in enumerate(loss_mean):
    ax.errorbar(np.arange(0, last), np.array(loss[0:last]),loss_std[i][0:last], \
                linewidth=linew, color=colors[i])
    plt.fill_between(np.arange(0, last), np.array(loss[0:last])-\
                     loss_std[i][0:last], np.array(loss[0:last])+\
                     loss_std[i][0:last], color=colors[i], alpha=0.1)
    plt.ylabel('Training loss', fontsize=font)
    plt.xlabel('Epoch', fontsize=font)
    legends.append('Label corruption=%.2f' % corruptions[i])
plt.legend(legends, loc='best', fontsize=font)
plt.xlim(0, last-1)
plt.ylim(0.1, 2.5)
ax.grid(color='gray', linestyle='dashdot', linewidth=1)
fig.savefig('pics/random_loss.png')
