import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.preprocessing import normalize

skewness = 50
x = stats.skewnorm.rvs(a=skewness, loc=100, size=1000)
print(f'min {x.min()}       max {x.max()}       mean {x.mean()}     median {np.median(x)}       mode {stats.mode(x)}')
# x1 = x.copy()
# x1[16:41] = input('Enter mode: ')
print(f'After amendment:\nmin {x.min()}       max {x.max()}       mean {x.mean()}     median {np.median(x)}       mode {stats.mode(x)}')
x2, x3, x4 = x.copy(), x.copy(), x.copy()

q2 = np.quantile(x, 0.5)
q3 = np.quantile(x, 0.75)
q1 = np.quantile(x, 0.25)
iqr = q3 - q1

for i in range(len(x)):
    if x[i] < q1 - 1.5*iqr or x[i] > q3 + 1.5*iqr:
        x2[i] = q2
        x3[i] = np.median(x)
        x4[i] = stats.mode(x)[0]
        
X = [x, x2, x3, x4]

fig, ax = plt.subplots(nrows=1, ncols=len(X), figsize=(18, 9))
r = 0
title = ['original x', 'replaced with mean', 'replaced with median', 'replaced with mode']
for i in X:
    ax[r].boxplot(i)
    ax[r].set_title(title[r])
    r += 1

plt.suptitle('Skewness : %d' %skewness)
plt.show()

pass