import pickle
import numpy as np
from matplotlib import pyplot as plt


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

result_indices_to_plot = np.array([0,1,3,4,5,6,7,8,9,10,11,13])

watermarkacc = np.array(metrics)[:, :, 2, 1][result_indices_to_plot]
watermarkacc = np.transpose(watermarkacc)

watermarkedacc = np.array(metrics)[:, :, 1, 1][result_indices_to_plot]
watermarkedacc = np.transpose(watermarkedacc)

cleanacc = np.array(metrics)[:, :, 0, 1][result_indices_to_plot]
cleanacc = np.transpose(cleanacc)

fig, ax = plt.subplots()
fig.set_size_inches(20,8)
ax.set_ylim(0,1)
ax.set_xlim(0,13)

showmeans = True
showextrema = False
showmedians = True

qs = calculate_quantiles(watermarkacc)

clean = plt.violinplot(
    cleanacc,
    showmeans=showmeans,
    showextrema=showextrema,
    showmedians=showmedians,
    quantiles=np.array([[.25, .75]]*len(result_indices_to_plot)).T,
    widths=.9)
watermarked = plt.violinplot(
    watermarkedacc,
    showmeans=showmeans,
    showextrema=showextrema,
    showmedians=showmedians,
    quantiles=np.array([[.25, .75]]*len(result_indices_to_plot)).T,
    widths=.9)
watermark = plt.violinplot(
    watermarkacc,
    showmeans=showmeans,
    showextrema=showextrema,
    showmedians=showmedians,
    quantiles=np.array([[.25, .75]]*len(result_indices_to_plot)).T,
    widths=.9)

# labels = np.array(['Debias',
#                    'Naive',
#                    'Unbalanced',
#                    'cleanmix',
#                    '10% counterexamples',
#                    'Matched',])
#
# plt.xticks(np.array([1, 2, 3, 4, 5, 6])[:len(result_indices_to_plot)],
#            labels[result_indices_to_plot])

plt.xticks(np.array(range(13))[:len(result_indices_to_plot)])

# plt.xlabel("Methods")
# plt.ylabel("Accuracy")
# plt.title("Auxiliary Watermark Accuracy")
# plt.legend([clean['bodies'][0], watermarked['bodies'][0], watermark['bodies'][0]],
#            ["$x_i$", "$x_i+n_i$", "$n_i$"])

plt.title('means medians quartiles')
plt.show()

print(f'Aux--: {np.average(watermarkacc, 0)}')
print(f'Water: {np.average(watermarkedacc, 0)}')
print(f'Clean: {np.average(cleanacc, 0)}')
