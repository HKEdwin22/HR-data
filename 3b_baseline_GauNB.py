# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from scipy import stats


def Visualize_Cont(x, f, c='mean'):
    '''
    Purpose: visualize the continuous data distribution
    x : dataframe
    f : feature
    c : mean/median/mode
    '''
    x1, x2, x3 = x[f].copy(), x[f].copy(), x[f].copy()

    q1 = np.quantile(x[f], 0.25)
    q2 = np.quantile(x[f], 0.5)
    q3 = np.quantile(x[f], 0.75)
    iqr = q3 - q1

    for i in range(len(x[f])):
        if x[f].iloc[i] < q1 - 1.5*iqr or x[f].iloc[i] > q3 + 1.5*iqr:
            x1[i] = np.mean(x[f])
            x2[i] = np.median(x[f])
            x3[i] = stats.mode(x[f])[0]

    X = [x[f], x1, x2, x3]

    _, ax = plt.subplots(nrows=1, ncols=len(X), figsize=(18, 9))
    title = ['original x', 'replaced with mean', 'replaced with median', 'replaced with mode']
    for r, i in enumerate(X):
        ax[r].boxplot(i)
        ax[r].set_title(title[r])

    plt.suptitle(f'Distribution of Feature {f}')
    plt.show()
    
    if c == 'mean':
        return x1
    elif c =='median':
        return x2
    else:
        return x3

def Outlier(x, f, t):
    '''
    Purpose: identify the outliers
    x : data set
    f : feature of the data set
    t : Cat or Cont
    '''
    if t == 'Cat':
        plt.pie(x[f].value_counts(), autopct='%1.2f%%')
        plt.title(f'Frequency of {f}')
    else:
        plt.boxplot(x[f])
        plt.title(f'Distribution of {f}')
    plt.get_current_fig_manager().full_screen_toggle()
    plt.savefig(f'{f}.png')
    plt.clf()

def Training(m, x, y):
    '''
    Purpose: Train the model
    m : Cat or Gau
    x : X
    y : target
    '''
    if m == 'Cat':
        clf = CategoricalNB(min_categories=x.nunique())
    else:
        clf = GaussianNB()
    X_tr, X_val, y_tr, y_val = train_test_split(x, y, test_size=0.2, random_state=41)

    clf.fit(X_tr, y_tr)
    p = clf.predict(X_val)

    acc_tr = accuracy_score(y_tr, clf.predict(X_tr))
    acc_val = accuracy_score(y_val, p)
    cm = confusion_matrix(y_val, p)

    plt.figure(figsize=(12,9))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Baseline Predication / (%.4f/%.4f)' %(acc_tr, acc_val))
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')

    plt.show()


# Load the data
df = pd.read_csv('./dataset2_cleaned_combined_dropOriginal.csv', index_col=0)

# Drop data that don't help
df = df.drop(['Zip'], axis=1)

# Split the data into categorical and continuous
X_cont = df[['Salary', 'Age']]
y = df['EmploymentStatus']

# Check outliers
New_X = []
for f in X_cont:
    New_X.append(Visualize_Cont(df, f))

X_cont = pd.DataFrame(New_X).transpose()
X_cont.columns = ['Salary', 'Age']

# Train the model
Training('Gau', X_cont, y)

pass