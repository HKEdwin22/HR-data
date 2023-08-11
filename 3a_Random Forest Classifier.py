# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.feature_selection import mutual_info_classif, SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import chi2_contingency


def Chi2_Independency(y, x, t=0.05):
    '''
    Purpose: Chi-sqaured test of independence
    Null hypothesis : the categories of the independent variables have the same effects on the dependent variable
    y : target variable
    x : feature
    t : threshold of p-value (default: 0.05)
    '''

    contingency_table = pd.crosstab(df[y], X_cat[x], margins=False)
    stat, p, dof, expected = chi2_contingency(contingency_table)

    # interpret p-value
    prob = 0.95
    alpha = 1.0 - prob
    print(f'-------------------{x}-------------------')
    if p <= alpha:
        print(f'p-value is {p:.4f}. Dependent (reject H0)')
    else:
        print(f'p-value is {p:.4f}. Independent (fail to reject H0)')


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

def Feature_Selection(x, y, m):
    '''
    Purpose: select features
    x : input dataset (X_cat)
    y : target
    m : MI or chi2
    '''
    plt.figure(figsize=(16,12))
    best_f = []
    score = []
    if m == 'MI':
        MI = mutual_info_classif(x, y, discrete_features=True, random_state=41)
        for scr, feature in sorted (zip(MI, x.columns), reverse=True):
            print(feature, round(scr, 4))
            score.append(scr)
            best_f.append(feature)
        plt.xticks(rotation=25)
    else:
        fs = SelectKBest(score_func=chi2, k='all')
        fs.fit(X_cat, y)
        # XX = fs.transform(X_cat)
        best_f = X_cat.columns[fs.get_support(indices=True)].tolist()
        score = fs.scores_
        for i in range(len(score)):
            print(f'Feature {i}     {best_f[i]}: {score[i]}')
        plt.xticks(rotation=80)

    plt.scatter(best_f, score)
    plt.title(f'Feature Selection ({m})')
    plt.xlabel('Feature')
    plt.ylabel('Score')
    plt.tight_layout(pad=5)
    plt.show()

    if m == 'chi2':
        return fs

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

    fig = plt.figure(figsize=(12,9))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Baseline Predication / (%.4f/%.4f)' %(acc_tr, acc_val))
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')

    plt.show()


# Load the data
df = pd.read_csv('./dataset2_cleaned_combined.csv', index_col=0)

# Drop data that don't help
df = df.drop(['Zip'], axis=1)

# Split the data into categorical and continuous
X_cont = df[['Salary', 'Age', 'ServiceYears']]
X_cat = df.drop(['Salary', 'Age', 'ServiceYears', 'EmploymentStatus'], axis=1)
y = df['EmploymentStatus']

# Baseline Model
# Training('Gau', X_cont, y)
# Training('Cat', X_cat, y)

# Check outliers
# for i in X_cat.columns:
#     Outlier(X_cat, i, 'Cat')
# for i in X_cont.columns:
#     Outlier(X_cont, i, 'Cont')

# Feature Selection with Random Forest
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_cat, y)
idx = rf.feature_importances_.argsort()
plt.figure(figsize=(16,9))
plt.barh(X_cat.columns[idx], rf.feature_importances_[idx])
plt.xlabel('Feature Importance')
plt.title('Feature Selection with Random Forests')
plt.tight_layout(pad=1)
plt.show()


pass