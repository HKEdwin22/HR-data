import matplotlib.pyplot as plt
import numpy as np

xC = [0.7702, 0.6667]
xG = [0.6734, 0.7460]

X_axis = np.arange(len(xC))
plt.bar(X_axis - 0.15, xC, 0.3, label='CategoricalNB')
plt.bar(X_axis + 0.15, xG, 0.3, label='GaussianNB')
plt.xticks(X_axis, ['Training', 'Validation'])
plt.legend(loc='center', bbox_to_anchor=(0.5,0.925))
plt.ylabel('Accuracy')
plt.xlabel('Set')
plt.title('Baseline Model Performance')

plt.show()