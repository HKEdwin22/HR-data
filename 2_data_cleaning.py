# Import libraries
from numba import jit, cuda
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
from timeit import default_timer as timer


def Check_Format(dataset, att, type):
    '''
    Purpose: Check the data format
    dataset : input dataframe
    att : an attribute a time
    type : num, str or date
    '''
    if type == 'num':
        dummy = dataset.applymap(lambda x: isinstance(x, (int, float)))[att]
        # print(att, dummy.value_counts()[True])
    elif type == 'str':
        dummy = dataset.applymap(lambda x: isinstance(x, (str)))[att]
        # print(att, dummy.value_counts()[True])
    elif type == 'date':
        if att == 'DOB':
            format = '%d/%m/%y'
            wrong = '%m/%d/%y'
            length = 8
        else:
            format = '%d/%m/%Y'
            wrong = '%m/%d/%Y'
            length = 10
        row = 0
        for i in dataset[att]:
            try:
                bool(datetime.strptime(i, format))                
            except ValueError as e:
                if bool(datetime.strptime(i, wrong)):
                    if len(i) == length:
                        m = i[:2]
                        d = i[3:5]
                        y = i[6:]
                    elif i[1] == '/':
                        m = '0' + i[0]
                        d = i[2:4]
                        y = i[5:]
                    elif i[4] == '/':
                        m = i[:2]
                        d = '0' + i[3]
                        y = i[5:]
                i = d + '/' + m + '/' + y
                dataset[att][row] = i
            row += 1

def Encode_Cat(x):
    '''
    Purpose: encoding categorical data
    x : input dataframe
    '''

    # Split the data set into categorical and numeric
    dob = x['DOB'].copy()
    EmpDate = x['DateofHire'].copy()
    cat = x.select_dtypes(include=['object']).copy()
    cat = cat.drop(['DOB', 'DateofHire'], axis=1)
    num = x.select_dtypes(include=['int']).copy()

    # Change dtype from object to category and encode
    for col in cat.columns:
        cat[col] = cat[col].astype('category')
        cat[col] = cat[col].cat.codes

    # Return an updated data set
    return pd.concat([cat, num, dob, EmpDate], axis=1, join='inner')

# @jit(target_backend='cuda')
def Visualize_Data(x):
    '''
    Purpose: Visualize the data distribution
    x : input dataframe
    '''
    start = timer()
    sns.pairplot(x, vars=['SpecialProjectsCount', 'Absences', 'EmpSatisfaction'], hue='EmploymentStatus')
    print(f'Plotting time: {round(timer() - start, 2)}s')
    plt.show()

def Useful_Date(x, att):
    duration = []
    if att == 'DOB':
        format = '%d/%m/%y'
    else:
        format = '%d/%m/%Y'
    today = datetime.today()
    # today = datetime.strptime(today, format)
        
    for i in x[att]:
        i = datetime.strptime(i, format)
        a = round((today - i).days / 365, 2)
        if a > 0:
            duration.append(a)
        else:
            duration.append(a + 100)

    if att == 'DOB':
        x['Age'] = duration
        x = x.drop([att], axis=1)
    else:
        x['ServiceYears'] = duration
        x = x.drop([att], axis=1)

    return x

    

if __name__ == '__main__':

    # Load the data
    df_raw = pd.read_csv('./dataset1_mined.csv', index_col=0)
    print('Shape of dataset: ', df_raw.shape)

    # Check empty cells by columns
    dummy = df_raw.isnull().sum()
    print('-------------Confirmed No Empty Cell in the Data Set-------------')

    # Check data format
    for a in df_raw.columns:
        if a == 'DOB' or a == 'DateofHire':
            t = 'date'
        elif a in ['Zip', 'EmpSatisfaction', 'SpecialProjectsCount', 'Absences', 'Salary']:
            t = 'num'
        else:
            t = 'str'
        Check_Format(df_raw, a, t)    
    print('-------------Data Format Checked-------------')    

    # Encode the features
    df_encoded = Encode_Cat(df_raw)
    print('-------------Categorical Features Encoded-------------')

    # Handle age and years of service
    df_2 = Useful_Date(df_encoded, 'DOB')
    df_2 = Useful_Date(df_2, 'DateofHire')
    del(df_encoded)
    
    # Check if the age is valid
    # row = 0
    # for i in df_2['Age']:
    #     if i > 65 or i < 0:
    #         print(f'False: row {row} : {i}')
    #     row += 1
    print('-------------Corrected Age and Service Year Computed-------------')

    plt.bar(df_2.iloc[:, 2])
    plt.show()

    # Data visualization
    print(df_raw.iloc[0])
    Visualize_Data(df_2)

    df_2.to_csv('dataset2_cleaned.csv')
    
    pass