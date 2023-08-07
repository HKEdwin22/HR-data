# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.feature_selection import mutual_info_classif


def Feature_Selection(x, y):
    '''
    Purpose: select features
    x : input dataset (X_cat)
    y : target
    '''
    best_f = []
    score = []
    MI = mutual_info_classif(x, y, random_state=0)
    for scr, feature in sorted (zip(MI, x.columns), reverse=True):
        print(feature, round(scr, 4))
        score.append(scr)
        best_f.append(feature)

    plt.figure(figsize=(16,12))
    plt.scatter(best_f, score)

    plt.title('MI of Features')
    plt.xticks(rotation=25)
    plt.xlabel('Feature')
    plt.ylabel('Mutual Information Score')
    plt.tight_layout(pad=5)

    plt.show()

def Training(m, x, y):
    '''
    Purpose: Train the model
    m : Cat or Gau
    x : X
    y : target
    '''
    if m == 'Cat':
        clf = CategoricalNB(min_categories=X_cat.nunique())
    else:
        clf = GaussianNB()
    X_tr, X_val, y_tr, y_val = train_test_split(x, y, test_size=0.2, random_state=41)

    clf.fit(X_tr, y_tr)
    p = clf.predict(X_val)

    acc_tr = accuracy_score(y_tr, clf.predict(X_tr))
    acc_val = accuracy_score(y_val, p)
    cm = confusion_matrix(y_val, p)

    fig = plt.figure(figsize=(12,9))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Baseline Predication / (%.4f/%.4f)' %(acc_tr, acc_val))
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')

    plt.show()


# Load the data
df = pd.read_csv('./dataset2_cleaned.csv', index_col=0)

# Drop data that don't help
df = df.drop(['Zip'], axis=1)

# Split the data into categorical and continuous
X_cont = df[['Salary', 'Age', 'ServiceYears']]
X_cat = df.drop(['Salary', 'Age', 'ServiceYears', 'EmploymentStatus'], axis=1)
y = df['EmploymentStatus']

# Baseline Model
Training('Gau', X_cont, y)
Training('Cat', X_cat, y)

# Feature Selection for CategoricalNB
Feature_Selection(X_cat, y)

pass