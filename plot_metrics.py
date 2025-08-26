import pickle
import numpy as np
from matplotlib import pyplot as plt

# File name here
f = open('metrics/full_vs_75_metric.pickle', 'rb')
metrics = pickle.load(f)
f.close()

# debias_quarters_P_match, watermark, debias_quarters, clean_mixed_marked_validation
# Metrics: method, repeat, test type(clean, watermarked, watermark), result type(loss, accuracy)

result_indices_to_plot = np.array([0,1,2])

watermarkacc = np.array(metrics)[:, :, 2, 1][result_indices_to_plot]
watermarkacc = np.transpose(watermarkacc)

watermarkedacc = np.array(metrics)[:, :, 1, 1][result_indices_to_plot]
watermarkedacc = np.transpose(watermarkedacc)

cleanacc = np.array(metrics)[:, :, 0, 1][result_indices_to_plot]
cleanacc = np.transpose(cleanacc)

plt.figure(figsize=(5, 4))

clean = plt.violinplot(cleanacc, showmeans=True, showextrema=True)
watermarked = plt.violinplot(watermarkedacc, showmeans=True, showextrema=True)
watermark = plt.violinplot(watermarkacc, showmeans=True, showextrema=True)

labels = np.array(['Naive',
                   '10% counterexamples',
                   'Matched',
                   'Naive',
                   '10% counterexamples',
                   'Matched',])

plt.xticks(np.array([1, 2, 3, 4, 5, 6])[:len(result_indices_to_plot)],
           labels[result_indices_to_plot])
plt.xlabel("Methods")
plt.ylabel("Accuracy")
# plt.title("Auxiliary Watermark Accuracy")
plt.legend([clean['bodies'][0], watermarked['bodies'][0], watermark['bodies'][0]],
           ["$x_i$", "$x_i+n_i$", "$n_i$"])
plt.show()

print(f'Aux--: {np.average(watermarkacc, 0)}')
print(f'Water: {np.average(watermarkedacc, 0)}')
print(f'Clean: {np.average(cleanacc, 0)}')
