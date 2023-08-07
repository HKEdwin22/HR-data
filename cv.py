from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

scr_tr, scr_val = [], []
for i in range(3,12,2):
    baseline_C = cross_validate(GauNB, X_cat, y, cv=i, n_jobs=4, return_train_score=True, error_score='raise')    
    baseline_G = cross_validate(GauNB, X_cat, y, cv=i, n_jobs=4, return_train_score=True, error_score='raise')
    tr = baseline_C['train_score'].mean()
    val = baseline_C['test_score'].mean()
    scr_tr.append(tr)
    scr_val.append(val)
    print(f'CV {i}      training accuracy: {tr:.4f}           validation accuracy: {val:.4f}')

plt.plot(range(3,12,2), scr_tr, label='training')
plt.plot(range(3,12,2), scr_val, label='validation')
plt.title('Accuracy against k-Fold Cross Validation')
plt.xticks([3,5,7,9,11])
plt.ylim([0.48,0.54])
plt.xlabel('k-Fold')
plt.ylabel('Accuracy')
plt.legend(loc='upper right', bbox_to_anchor=(.985,.98))