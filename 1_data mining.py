# Import libraries
import pandas as pd


# Study the data
def Feature_Info(x):
    '''
    Purpose: First look of features
    x : input dataframe
    '''
    dtype = []

    for i in x.columns:
        t = x[i].dtype
        if t not in dtype:
            dtype.append(t)

    return dtype



if __name__ == '__main__':
    # Load the data
    df_raw = pd.read_csv('./HRDataset_v14.csv')
    
    data_tp = Feature_Info(df_raw)
    # print(df.select_dtypes(include='float64').head())
    
    # Remove the encoded features and EmpID
    df = df_raw.drop(['EmpID', 'MarriedID', 'MaritalStatusID', 'GenderID', 'EmpStatusID', 'DeptID', 'PerfScoreID', 'FromDiversityJobFairID', 'Termd', 'PositionID', 'ManagerID'], axis=1)
    # Remove imbalanced and unclear features
    df = df.drop(['HispanicLatino', 'CitizenDesc', 'TermReason', 'LastPerformanceReview_Date', 'DaysLateLast30', 'DateofTermination', 'EngagementSurvey'], axis=1)
    # Remove unused features
    df = df.drop(['Employee_Name'], axis=1)

    df.to_csv('dataset_1.csv')

pass