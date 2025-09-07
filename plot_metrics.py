import pickle
import numpy as np
from matplotlib import pyplot as plt
from sympy.printing.pretty.pretty_symbology import line_width


def calculate_quantiles(group):
    return np.quantile(group, [.25, .75], axis=0)


metrics = []

filenames = [
    'metrics/metrics_debias_naive_unbalanced_cleanmix.pickle',      # 0,1,2,3
    'metrics/full_vs_75_metric.pickle',                             # 4,5,6,7,8,9
    'metrics/quarters_vs_halves_metrics.pickle',                    # 10,11
    'metrics/quarters_true_vs_false_metrics.pickle'                 # 12,13
]

for name in filenames:
    f = open(name, 'rb')
    metrics += pickle.load(f)
    f.close()

# # File name here
# f = open('metrics/THIS_FILE_DOES_NOT_EXIST', 'rb')
# metrics = pickle.load(f)
# f.close()


# Metrics: method, repeat, test type(clean, watermarked, watermark), result type(loss, accuracy)
result_indices_to_plot = np.array([0,1,3,4,5,6,7,8,9,10,13])

watermarkacc = np.array(metrics)[:, :, 2, 1][result_indices_to_plot]
watermarkacc = np.transpose(watermarkacc)

watermarkedacc = np.array(metrics)[:, :, 1, 1][result_indices_to_plot]
watermarkedacc = np.transpose(watermarkedacc)

cleanacc = np.array(metrics)[:, :, 0, 1][result_indices_to_plot]
cleanacc = np.transpose(cleanacc)

# Plot setup
fontsize = 19

fig, ax = plt.subplots()
fig.set_size_inches(17,8)
plt.subplots_adjust(bottom=0.25,left=0.055,top=0.95,right=0.997)

ax.set_ylim(0,1)
ax.set_xlim(.5,len(result_indices_to_plot)+.5)

ax.spines[['right', 'top']].set_visible(False)

plt.grid(which="major",axis='y', alpha=.4, color='black', linewidth=1)
plt.grid(which="minor",axis='y', alpha=.2, color='black')

showmeans = False
showextrema = False
showmedians = True

# Do plot

qs = calculate_quantiles(watermarkacc)

clean = plt.violinplot(
    cleanacc,
    showmeans=showmeans,
    showextrema=showextrema,
    showmedians=showmedians,
    quantiles=np.array([[.25, .75]]*len(result_indices_to_plot)).T,
    widths=.95)
watermarked = plt.violinplot(
    watermarkedacc,
    showmeans=showmeans,
    showextrema=showextrema,
    showmedians=showmedians,
    quantiles=np.array([[.25, .75]]*len(result_indices_to_plot)).T,
    widths=.95)
watermark = plt.violinplot(
    watermarkacc,
    showmeans=showmeans,
    showextrema=showextrema,
    showmedians=showmedians,
    quantiles=np.array([[.25, .75]]*len(result_indices_to_plot)).T,
    widths=.95)

# Separators
lines = [
    1.5,
    3.5,
    6.5,
    9.5,
    10.5
]
for line in lines:
    plt.axvline(x=line, color='black', alpha=.4, linewidth=1, linestyle=(0, (20, 20)))

# Labels

plt.rcParams['text.usetex'] = True

ax.tick_params(labelsize=fontsize)

labels = np.array(['Ours:\n$X_i+N_i$\n$(X_i+N_i) \oplus C_{t_{1 \dots I-1}}$\n$C_i$',
                   'Naive:\n$X_i+N_i$',
                   'Imbalanced:\n$X_i+N_i$\n$(X_i+N_i) \oplus C_j$\n(unbalanced)\n$C_i$',
                   '$X_i+N_i$\n$C_i$',
                   'Naive\n75%\nwatermark',
                   '10%\nexamples\n75%\nwatermark',
                   'Ours\n75%\nwatermark',
                   'Naive\n100%\nwatermark',
                   '10%\nexamples\n100%\nwatermark',
                   'Ours\n100%\nwatermark',
                   '1/2 size\npatches',
                   'DO NOT USE\nredundant\nOurs',
                   'DO NOT USE\nredundant\nOurs',
                   'Ours\nscrambled\ntesting\nwatermarks',
                   ])

plt.rcParams['text.usetex'] = False

plt.xticks(np.array(range(1,16))[:len(result_indices_to_plot)],
           labels[result_indices_to_plot])

plt.yticks(np.linspace(0,1,10, endpoint=False), minor=False)
plt.yticks(np.linspace(0,1,20, endpoint=False), minor=True)

# plt.xticks(np.array(range(13))[:len(result_indices_to_plot)+1])

plt.xlabel("Methods", fontsize=fontsize)
plt.ylabel("Accuracy", fontsize=fontsize)
# plt.title("Auxiliary Watermark Accuracy")
plt.legend([clean['bodies'][0], watermarked['bodies'][0], watermark['bodies'][0]],
           ["$X_i$", "$X_i+N_i$", "$N_i$"], fontsize=fontsize, loc='upper right')

plt.title('Experimental results', fontsize=fontsize)

# Finish

plt.savefig("all_experiments.pdf", format="pdf")
plt.show()

# plt.savefig("plot.png", format="png")

print(f'Aux--: {np.average(watermarkacc, 0)}')
print(f'Water: {np.average(watermarkedacc, 0)}')
print(f'Clean: {np.average(cleanacc, 0)}')

