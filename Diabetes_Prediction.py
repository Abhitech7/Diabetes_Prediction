# @AUTHOR : ARDA BAYSALLAR
# DIABETES PREDICTION

# MAIN TARGET TO DETECT DIABETES PEOPLE USING THEIR INFORMATION

# DATA SET STORY:
# The dataset is part of the large dataset held at the National
# Institutes of Diabetes-Digestive-Kidney Diseases in the USA.
# Data used for diabetes research on Pima Indian women aged 21 and
# over living in Phoenix, the 5th largest city of the State of Arizona
# in the USA.
# The target variable is specified as "outcome"; 1 indicates positive
# diabetes test result, 0 indicates negative.

# FEATURES :
# Pregnancies: Number of pregnancies
# Glucose: 2-hour plasma glucose concentration in the oral glucose tolerance test
# Blood Pressure : Blood Pressure (mm Hg)
# SkinThickness : Skin Thickness
# Insulin: 2-hour serum insulin (mu U/ml)
# DiabetesPedigreeFunction : A function that calculates the probability of having diabetes based on the pedigree
# BMI : Body mass index
# Age : Age (years)
# Outcome: Have the disease (1) or not (0)
########################################################################################
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

########################################################################################
# Read the dataset
########################################################################################
df_main = pd.read_csv('diabetes.csv')
df=df_main.copy()
df.head()
########################################################################################
# EDA
########################################################################################
df.info()
df.describe().T
########################################################################################
# IMPORTANT NOTES --------------------------------------------------------------------------------------------------
"""
* blood pressure = 0 : 
Low blood pressure is generally considered a blood pressure reading 
    lower than 90 millimeters of mercury (mm Hg) for the top number (systolic) 
    or 60 mm Hg for the bottom number (diastolic)
    reference : https://www.mayoclinic.org/diseases-conditions/low-blood-pressure/symptoms-causes/syc-20355465#:~:text=Low%20blood%20pressure%20is%20generally,the%20bottom%20number%20(diastolic).

* skin thickness = 0 : 
    (25 to 40 μm2) 
    reference : https://en.wikipedia.org/wiki/Human_skin#:~:text=A%20skin%20cell%20usually%20ranges,the%20dermis%20and%20the%20hypodermis.

* Insulin = 0 : 
    Reference Range 16-166 mIU/L 
    reference : https://emedicine.medscape.com/article/2089224-overview

* BMI = 0 : 
    underweight (under 18.5 kg/m2), 
    normal weight (18.5 to 24.9), 
    overweight (25 to 29.9), 
    and obese (30 or more) 
    reference : https://en.wikipedia.org/wiki/Body_mass_index#:~:text=Major%20adult%20BMI%20classifications%20are,obese%20(30%20or%20more).

* Glucose = 0 :
    (Two hours after drinking the glucose solution, a normal blood 
    glucose level is lower than 155 mg/dL (8.6 mmol/L) - 8 saat açlık 110-126 
    
    reference: https://www.mayoclinic.org/tests-procedures/glucose-tolerance-test/about/pac-20394296#:~:text=A%20normal%20fasting%20blood%20glucose,(8.6%20mmol%2FL).

According to these references zero minimum value can not be possible if the subject is not dead
"""
# ****-------------------------------------------------------------------------------------------------------------------------------------
########################################################################################
df.isna().sum()
pd.isna(df).any()

#zero_value_indexes = df.iloc[np.where(df[num_cols[1:]]==0)[0]].index
#zero_value_indexes = zero_value_indexes.unique()
must_non_zero_cols = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']


# create supposed to be null values
for col in must_non_zero_cols:
    df.loc[df[col] == 0.0, col] = np.nan


def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) |
                 (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


Reference_Ranges = {'Glucose':[1,155],
                    'BloodPressure':[60,90],
                    'SkinThickness':[25,40],
                    'BMI':[18.5,25],
                    'Insulin':[16,166]}


def check_anomaly(dataframe=df,reference=Reference_Ranges):
    for col in reference.keys():
        if ((dataframe[col].min() >= reference[col][0]) &
            (dataframe[col].max() <= reference[col][1])):
            print(col,' : all in normal range' )
        else :
            print(col,' : ALERT')

df.Pregnancies.value_counts() # this should be considered as categorical

########################################################################################
# GATHER NUMERIC AND CATEGORICAL FEATURES
########################################################################################

def grab_col_names(dataframe, cat_th=18, car_th=30):
    """

    It gives the names of categorical, numeric and categorical but ordinal variables in the data set.
    Note: Categorical variables with numerical appearance are also included in categorical variables.

    Parameters
    ------
        dataframe: dataframe
                The dataframe from which variable names are to be retrieved
        cat_th: int, optional
                class threshold for numeric but categorical variables
        car_th: int, optinal
                class threshold for categorical but cardinal variables

    Returns
    ------
        cat_cols: list
                Categorical features
        num_cols: list
                Numeric features
        cat_but_car: list
                Cardinal Features

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = total number of features
        num_but_cat is in cat_cols.
        Return The sum of the 3 lists is equal to the total number of variables:
            cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

for col in cat_cols:
    print(df.loc[:,col].value_counts())

df.loc[:,num_cols].describe().T

########################################################################################
# TARGET VARIABLE ANALYSIS
########################################################################################

def target_variable_analysis (df):
    print('CATEGORICAL COLUMNS', '\n',
          df.groupby(cat_cols).agg({'Outcome':'mean'}), '\n\n')
    print('NUMERICAL COLUMNS', '\n',
          df.pivot_table(values=num_cols, columns='Outcome', aggfunc='mean'), '\n\n')

target_variable_analysis(df)

########################################################################################
# OUTLIER ANALYSIS :
########################################################################################

outlier_cols = []
for col in num_cols:
    print(col, check_outlier(df, col))
    if check_outlier(df, col):
        outlier_cols.append(col)

########################################################################################
# Clipping 1% OUTLIERS FROM BOTH SIDE
########################################################################################

df[outlier_cols]

# remove outlier %1 from both side
def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) |
                                      (dataframe[col_name] > up_limit))]
    return df_without_outliers

for col in num_cols:
    new_df = remove_outlier(df, col)

# check outlier cols after removal
for col in outlier_cols:
    print(col, '-->', check_outlier(new_df, col))

check_anomaly(dataframe=df)
new_df.head()
new_df.isna().sum()
#######################################################################################
# Local Outlier Factor
########################################################################################
def LOF_Maker( dataframe,do=False):
    new_df = dataframe
    clf = LocalOutlierFactor(n_neighbors=25)
    clf.fit_predict(new_df.dropna())

    df_scores = clf.negative_outlier_factor_
    df_scores[0:5]
    # df_scores = -df_scores
    np.sort(df_scores)[0:5]

    scores = pd.DataFrame(np.sort(df_scores), columns=['score'])
    scores.plot(stacked=True, xlim=[0, 50], style='.-')
    plt.show()
    scores.head()
    th = np.sort(df_scores)[25]
    print(th)
#LOF_Maker(new_df,do=True)

# we have too many missing value
#df[df_scores<th]
#df[df_scores < th].shape
#df.describe([0.01, 0.05, 0.75, 0.90, 0.99]).T
#df[df_scores < th].index
#df[df_scores < th].drop(axis=0, labels=df[df_scores < th].index)

########################################################################################
# MISSING VALUE ANALYSIS
########################################################################################
msno.bar(new_df)
plt.show()
msno.matrix(new_df)
plt.show()
# missing correlation
msno.heatmap(new_df)
plt.show()

########################################################################################
# CORR ANALYSIS
########################################################################################

def corr_matrix_plot(dataframe, type=True):
    import plotly.express as px
    import plotly.io as pio
    pio.renderers.default = "browser"

    corr_matrx = new_df.corr()
    # web heatmap
    if type:
        fig = px.imshow(corr_matrx, text_auto=True)
        fig.show()
    # alternative heatmap
    else:
        sns.heatmap(corr_matrx, annot=True)
        plt.show()
corr_matrix_plot(new_df)

def high_corr_column_grabber(corr_matrx, dataframe):
    high_corrs= {}
    for col in corr_matrx.columns:
        high_corr = np.where(corr_matrx[col] > 0.6)
        if high_corr[0].size > 1 :
            col1 = dataframe.columns[high_corr[0][0]]
            col2 = dataframe.columns[high_corr[0][1]]
            if col1 not in high_corrs.items():
                high_corrs[col1] = [col2]
            else :
                high_corrs[col1].append(col2)

    return high_corrs

hcorrs = high_corr_column_grabber(corr_matrx, new_df)
print(hcorrs)

########################################################################################
# FEATURE ENGINEERING
########################################################################################
# Take necessary actions for missing and outlier values. There are no missing
# observations in the data set, but Glucose, Insulin etc.
# Observation units containing a value of 0 in the variables may
# represent the missing value. For example; a person's glucose or insulin value
# will not be 0. Considering this situation, you can assign the zero values to
# the relevant values as NaN and then apply the operations to the missing values.
new_df.isnull().sum()

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns
missing_values_table(new_df)
na_cols = missing_values_table(new_df, True)

def missing_histograms(dataframe, na_cols):
    count=0
    fig, axs = plt.subplots(len(na_cols))
    fig.suptitle('HISTOGRAMS')
    for col in na_cols :
        axs[count].hist(dataframe[col])
        count += 1
    plt.show()
missing_histograms(new_df, na_cols)

missing_histograms(new_df, na_cols)
missing_histograms(new_df.fillna(method='backfill',axis=0), na_cols)
missing_histograms(new_df.interpolate(), na_cols)
missing_histograms(new_df.interpolate(method='nearest'), na_cols)

new_df.interpolate(method='nearest', inplace=True)
new_df.dropna(inplace=True)

################################################################################
# NEW FEATURES:
################################################################################

Reference_Ranges
# non frequentist approach

for k, v in Reference_Ranges.items():
    print('Dataframe: ',new_df[k].mean(), '---- Reference:', v )
    print(k, '-->', -new_df[k].mean() + ((v[1]-v[0])+v[0]), end='\n\n')

#############################################################
# Feature Engineering : BMI
#############################################################
# BMI
# underweight (under 18.5 kg/m2),
# normal weight (18.5 to 24.9),
# overweight (25 to 29.9),
# obese (30 or more)

new_df['BMI_CAT'] = pd.cut(new_df.BMI,
                           [new_df.BMI.min(), 18.5, 25, 30, new_df.BMI.max()],
                           labels=['underweight', 'normal', 'overweight', 'obese'],
                           include_lowest=True)

#############################################################
# Feature Engineering : INSULIN
#############################################################

# Insulin upper bound is ranged to 240 or up

new_df['Insulin_CAT'] = pd.cut(new_df.Insulin,
                           [new_df.Insulin.min(), 110, 150, 240, new_df.Insulin.max()],
                           labels=['Low', 'normal', 'high', 'extreme'],
                           include_lowest=True)

low_limit, up_limit = outlier_thresholds(new_df, 'Insulin', q1=0.3, q3=0.7)
new_df.loc[new_df['Insulin'] > up_limit,'Insulin'] = (up_limit +
                                                     240*(new_df.loc[new_df['Insulin'] > up_limit,'Insulin'] /
                                                     new_df.Insulin.max()))


new_df.head(10)
df.iloc[new_df.index,:].head(10)

new_df.describe().T

#############################################################
# Feature Engineering : SKIN THICKNESS
#############################################################

# Low : <25
# Normal : 25-40
# High : 40 >

new_df['SkinThickness_CAT'] = pd.cut(new_df.SkinThickness,
                                     [new_df.SkinThickness.min(), 25, 40, new_df.SkinThickness.max()],
                                     labels=['Low', 'normal', 'high'],
                                     include_lowest=True)

new_df.head(10)
df.iloc[new_df.index,:].head(10)


#############################################################
# Feature Engineering : BLOOD PRESSURE
#############################################################
# Suggested : https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2245318/pdf/procascamc00018-0276.pdf
# Low : <76.2
# Normal : 76.2-98.1
# High : > 98.1

new_df['BloodPressure_CAT'] = pd.cut(new_df.BloodPressure,
                                     [new_df.BloodPressure.min(), 76.2, 98.1, new_df.BloodPressure.max()],
                                     labels=['Low', 'normal', 'high'],
                                     include_lowest=True)

new_df.head(10)
df.iloc[new_df.index,:].head(10)

#############################################################
# Feature Engineering : DiabetesPedigreeFunction
#############################################################
# Suggested : https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2245318/pdf/procascamc00018-0276.pdf
# Low : < 0.245
# Low-Normal : 0.245 - 0.525
# Normal : 0.526 - 0.805
# High-Normal : 0.806 - 1.111
# High : 1.111 +

new_df['DiabetesPedigreeFunction_CAT'] = pd.cut(new_df.DiabetesPedigreeFunction,
                                     [new_df.DiabetesPedigreeFunction.min(),
                                      0.245, 0.526, 0.806, 1.111,
                                      new_df.DiabetesPedigreeFunction.max()],
                                     labels=['Low', 'Low-Normal', 'Normal', 'High-Normal','High'],
                                     include_lowest=True)

new_df.head(10)


#############################################################
# Feature Engineering : Glucose
#############################################################
# Suggested : https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2245318/pdf/procascamc00018-0276.pdf
# Low : < 89.1
# Low-Normal : 89.1 - 107.1
# Normal : 107.2 - 123.1
# High-Normal : 123.2 - 143.1
# High : 143.2 - 165.1
# Extreme : 165.2 +


new_df['Glucose_CAT'] = pd.cut(new_df.Glucose,
                               [new_df.Glucose.min(),
                                89.1, 107.1, 123.1, 143.1, 165.1,
                                new_df.Glucose.max()],
                               labels=['Low', 'Low-Normal', 'Normal', 'High-Normal', 'High', 'Extreme'],
                               include_lowest=True)
new_df.head(10)

#############################################################
# Feature Engineering : AGE
#############################################################

new_df['Age_CAT'] = pd.qcut(new_df.Age,q=[0.0, 0.25, 0.5, 0.75, 1], labels=[1, 2, 3, 4])
new_df.head(10)



########################################################################################
# ENCODING :
########################################################################################
####################
# ordinal or binary
####################
def label_encoder(dataframe, col):
    labelencoder = LabelEncoder()
    dataframe[col] = labelencoder.fit_transform(dataframe[col])
    return dataframe
new_df.info()

new_df = label_encoder(new_df,'Age_CAT')
new_df = label_encoder(new_df,'DiabetesPedigreeFunction_CAT')
new_df = label_encoder(new_df,'Glucose_CAT')
new_df = label_encoder(new_df,'SkinThickness_CAT')
new_df = label_encoder(new_df,'Insulin_CAT')
new_df = label_encoder(new_df,'BMI_CAT')
new_df = label_encoder(new_df,'BloodPressure_CAT')
new_df.head()

####################
# rare encoding
####################
def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(new_df, "Outcome", ['Pregnancies'])

def rare_encoder(dataframe, rare_perc, rare_cols):
    temp_df = dataframe.copy()

    rare_columns = rare_cols
    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df

new_df = rare_encoder(new_df, 0.014, ['Pregnancies'])
new_df.Pregnancies.value_counts()
new_df = label_encoder(new_df,'Pregnancies')
new_df.Pregnancies.value_counts()

new_df.head()
#############################################
# Feature Scaling
#############################################

new_df.hist()
plt.show()

scale_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
Scaler = RobustScaler()

new_df[scale_cols] = Scaler.fit_transform(new_df[scale_cols])
new_df.describe().T


#############################################
# MODEL
#############################################
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import binarize
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, roc_auc_score, precision_recall_curve
from sklearn.metrics import f1_score, fbeta_score, accuracy_score, classification_report, average_precision_score
from sklearn import model_selection
import warnings
warnings.filterwarnings("ignore") # Don't want to see the warnings in the notebook

df = new_df.copy()
seed = 42 # for reproducibility
train, test = model_selection.train_test_split(df, test_size=0.30, random_state=seed, stratify=df['Outcome'])
y_train = train['Outcome']
X_train = train.drop('Outcome', axis=1)
y_test = test['Outcome']
X_test = test.drop('Outcome', axis=1)
print('X_train and y_train:', X_train.shape, y_train.shape)
print('X_test and y_test  :', X_test.shape, y_test.shape)

##############################################################
# MODEL BUILDING : LOGISTIC REGRESSION CLASSIFICATION
##############################################################

pipeline = Pipeline(steps=[('classifier', LogisticRegression(random_state=seed))])
params = [{'classifier__C': np.arange(0.1, 2.0, 0.05),
           'classifier__penalty': ['l1'],
           'classifier__solver': ['liblinear', 'saga']},

          {'classifier__C': np.arange(0.1, 2.0, 0.05),
           'classifier__penalty': ['l2'],
           'classifier__solver': ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga']}
          ]
kfold = StratifiedKFold(n_splits=10, random_state=None)  # train/validation with the same ratio of classes
grid = GridSearchCV(pipeline, param_grid=params, cv=kfold, verbose=1, n_jobs=-1)
grid.fit(X_train, y_train)
print(grid.best_estimator_,'\n')


# BEST PARAMS
print('Best parameters  :', grid.best_params_)
print('\nTraining accuracy:', grid.score(X_train, y_train))
print('Test accuracy    :', grid.score(X_test, y_test))


# CONFUSION MATRIX
def draw_cm(actual, predicted):
    cm = confusion_matrix(actual, predicted)
    sns.heatmap(cm, annot=True,  fmt='.0f', xticklabels = ["Outcome:1", "Outcome:0"] ,
                yticklabels = ["Outcome:1", "Outcome:0"] )
    plt.ylabel('ACTUAL')
    plt.xlabel('PREDICTED')
    plt.show()
draw_cm( y_test, grid.predict(X_test) )
print('\n',classification_report(y_test, grid.predict(X_test)))

# PROBS :
probs = grid.predict_proba(X_test)
data = {'Actual'   : y_test,
        'Predicted': grid.predict(X_test),
        'Prob(0)'  : probs[:, 0],
        'Prob(1)'  : probs[:, 1]
        }

dfprobs = pd.DataFrame (data)
dfprobs.sample(5)

# PROBS ANALYSIS
y0 = dfprobs[dfprobs.Actual == 0]['Prob(1)']
y1 = dfprobs[dfprobs.Actual == 1]['Prob(1)']
plt.figure(figsize=(16,4))
plt.subplot(121)
plt.hist(y0, bins=20, color='green', alpha=0.4, label="Outcome:0 (actual)")
plt.hist(y1, bins=20, color='red', alpha=0.4, label="Outcome:1 (actual)")
plt.axvline(x=0.5, linestyle='--', color='k')
plt.title('Histogram: Predicted probabilities')
plt.legend()

plt.subplot(122)
sns.kdeplot(y1, color="red", shade=True, label="Outcome:1 (actual)")
sns.kdeplot(y0, color="green", shade=True, label="Outcome:0 (actual)")
plt.axvline(x=0.5, linestyle='--', color='k')
plt.annotate('TP', ha='center', va='bottom', size=14, xytext=(0.8,2),
             xy=(0.6,0.6), arrowprops={'facecolor':'red', 'shrink':0.01})
plt.annotate('FP', ha='center', va='bottom', size=14, xytext=(0.9,1.4),
             xy=(0.7,0.05), arrowprops={'facecolor':'green', 'shrink':0.01})
plt.annotate('TN', ha='center', va='bottom', size=14, xytext=(0.2,5),
             xy=(0.1,4), arrowprops={'facecolor':'green', 'shrink':0.01})
plt.annotate('FN', ha='center', va='bottom', size=14, xytext=(0.3,3),
             xy=(0.3,2), arrowprops={'facecolor':'red', 'shrink':0.01})
plt.title('Density estimate: Predicted probabilities')
plt.show()

"""
The graph above shows us the decision boundary at p=0.5 (black vertical line). 
It divides the whole region into 4 sections where Pure green is Outcome 0 : not diabetic, 
and Pure red is Outcome 1 : Diabetic :

Green area to the right of decision boundary: False positives
Red area to the right of decision boundary: True Positives
Green area to the left of decision boundary: True Negatives
Red to area the left of decision boundary: False Negatives
"""


# ROC CURVE :
cm = confusion_matrix(y_test, grid.predict(X_test))

# Determine the false positive and true positive rates for ROC
fpr, tpr, _ = roc_curve(y_test, grid.predict_proba(X_test)[:,1])
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label='LR (ROC-AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
tn, fp, fn, tp = [i for i in cm.ravel()]
plt.plot(fp/(fp+tn), tp/(tp+fn), 'ro', markersize=8, label='Decision Point')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (TPR vs FPR at each probability threshold)')
plt.legend(loc="lower right")
plt.show()

##############################################################
# MODEL BUILDING : RANDOM FOREST
##############################################################


pipeline = Pipeline(steps=[('classifier', RF(random_state=seed))])
params = [{'classifier__criterion': ['gini','entropy','log_loss'],
           'classifier__n_estimators': [50,100,150,200,250,300],
           'classifier__max_depth': [5, 7, 10, 13],
            'classifier__min_samples_split': [2,3,4],
            'classifier__max_features': [8,10,12],
            'classifier__min_samples_leaf': [1,2,3]
           },
          ]
kfold = StratifiedKFold(n_splits=10, random_state=None)  # train/validation with the same ratio of classes
grid = GridSearchCV(pipeline, param_grid=params, cv=kfold, verbose=1, n_jobs=-1)
grid.fit(X_train, y_train)
print(grid.best_estimator_,'\n')

print('Best parameters  :', grid.best_params_)
print('\nTraining accuracy:', grid.score(X_train, y_train))
print('Test accuracy    :', grid.score(X_test, y_test))
