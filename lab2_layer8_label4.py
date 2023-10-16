# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T09:45:31.738940Z","iopub.execute_input":"2023-09-24T09:45:31.739341Z","iopub.status.idle":"2023-09-24T09:45:31.749487Z","shell.execute_reply.started":"2023-09-24T09:45:31.739310Z","shell.execute_reply":"2023-09-24T09:45:31.748009Z"}}
import pandas as pd
import numpy as np
from pandas import Series

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
import seaborn as sns

import matplotlib.pyplot as plt
%matplotlib inline

import warnings
warnings.filterwarnings("ignore")

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T09:45:33.138102Z","iopub.execute_input":"2023-09-24T09:45:33.139904Z","iopub.status.idle":"2023-09-24T09:45:41.080585Z","shell.execute_reply.started":"2023-09-24T09:45:33.139846Z","shell.execute_reply":"2023-09-24T09:45:41.079225Z"}}
train_data_frame = pd.read_csv('/kaggle/input/dataset-lab2/train.csv')
valid_data_frame = pd.read_csv('/kaggle/input/dataset-lab2/valid.csv')
test_data_frame = pd.read_csv('/kaggle/input/dataset-lab2/test.csv')

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T09:45:41.083180Z","iopub.execute_input":"2023-09-24T09:45:41.083696Z","iopub.status.idle":"2023-09-24T09:45:41.092910Z","shell.execute_reply.started":"2023-09-24T09:45:41.083640Z","shell.execute_reply":"2023-09-24T09:45:41.091466Z"}}
train_data_frame.shape

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T09:45:41.094438Z","iopub.execute_input":"2023-09-24T09:45:41.094822Z","iopub.status.idle":"2023-09-24T09:45:41.132153Z","shell.execute_reply.started":"2023-09-24T09:45:41.094788Z","shell.execute_reply":"2023-09-24T09:45:41.130719Z"}}
train_data_frame.head()

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T09:45:41.134630Z","iopub.execute_input":"2023-09-24T09:45:41.134989Z","iopub.status.idle":"2023-09-24T09:45:41.175542Z","shell.execute_reply.started":"2023-09-24T09:45:41.134958Z","shell.execute_reply":"2023-09-24T09:45:41.174656Z"}}
missing_cols = train_data_frame.columns[train_data_frame.isnull().any()]
missing_counts = train_data_frame[missing_cols].isnull().sum()

print('Missing Columns and Counts')
for column in missing_cols:
    print( str(column) +' : '+ str(missing_counts[column]))

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T09:45:43.965576Z","iopub.execute_input":"2023-09-24T09:45:43.966049Z","iopub.status.idle":"2023-09-24T09:45:44.064294Z","shell.execute_reply.started":"2023-09-24T09:45:43.966011Z","shell.execute_reply":"2023-09-24T09:45:44.063033Z"}}
train_data = train_data_frame.copy()
valid_data = valid_data_frame.copy()
test_data = test_data_frame.copy()

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T09:45:46.250641Z","iopub.execute_input":"2023-09-24T09:45:46.251038Z","iopub.status.idle":"2023-09-24T09:45:49.250243Z","shell.execute_reply.started":"2023-09-24T09:45:46.251007Z","shell.execute_reply":"2023-09-24T09:45:49.249007Z"}}
train_data_frame.describe()

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T09:45:49.252739Z","iopub.execute_input":"2023-09-24T09:45:49.253681Z","iopub.status.idle":"2023-09-24T09:45:54.943639Z","shell.execute_reply.started":"2023-09-24T09:45:49.253629Z","shell.execute_reply":"2023-09-24T09:45:54.942692Z"}}
from sklearn.preprocessing import RobustScaler # eliminate outliers

Xtrain = {}
ytrain = {}

Xvalid = {}
yvalid = {}

Xtest = {}
ytest = {}

#create dictionaries for each label
for target_label in ['label_1','label_2','label_3','label_4']:

  if target_label == "label_2":
    train = train_data_frame[train_data_frame['label_2'].notna()]
    valid = valid_data_frame[valid_data_frame['label_2'].notna()]
  else:
    train = train_data_frame
    valid = valid_data_frame

  test = test_data_frame

  scaler = RobustScaler()

  Xtrain[target_label] = pd.DataFrame(scaler.fit_transform(train.drop(['label_1','label_2','label_3','label_4'], axis=1)), columns=[f'feature_{i}' for i in range(1,769)])
  ytrain[target_label] = train[target_label]

  Xvalid[target_label] = pd.DataFrame(scaler.transform(valid.drop(['label_1','label_2','label_3','label_4'], axis=1)), columns=[f'feature_{i}' for i in range(1,769)])
  yvalid  [target_label] = valid[target_label]

  Xtest[target_label] = pd.DataFrame(scaler.transform(test.drop(["ID"],axis=1)), columns=[f'feature_{i}' for i in range(1,769)])

# %% [code]
Xtrain_data_frame = Xtrain['label_4'].copy()
ytrain_data_frame = ytrain['label_4'].copy()

Xvalid_data_frame = Xvalid['label_4'].copy()
yvalid_data_frame = yvalid['label_4'].copy()

Xtest_data_frame = Xtest['label_4'].copy()

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T09:49:55.079334Z","iopub.execute_input":"2023-09-24T09:49:55.079818Z","iopub.status.idle":"2023-09-24T10:11:56.737941Z","shell.execute_reply.started":"2023-09-24T09:49:55.079781Z","shell.execute_reply":"2023-09-24T10:11:56.736696Z"}}
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, KFold

# Perform cross-validation
scores = cross_val_score(SVC(), Xtrain_data_frame, ytrain_data_frame, cv=5, scoring='accuracy')

mean_accuracy = scores.mean()
std_accuracy = scores.std()

# Print the cross-validation scores
print('Support Vector Machines')
print('\n')
print("Cross-validation scores:", scores)
print(f"Mean Accuracy: {mean_accuracy:.2f}")
print(f"Standard Deviation: {std_accuracy:.2f}")

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T10:12:50.999387Z","iopub.execute_input":"2023-09-24T10:12:50.999989Z","iopub.status.idle":"2023-09-24T10:12:55.401706Z","shell.execute_reply.started":"2023-09-24T10:12:50.999942Z","shell.execute_reply":"2023-09-24T10:12:55.400477Z"}}
from sklearn.decomposition import PCA

pca = PCA(n_components=0.975, svd_solver='full')
pca.fit(Xtrain_data_frame)
Xtrain_data_frame_pca = pd.DataFrame(pca.transform(Xtrain_data_frame))
Xvalid_data_frame_pca = pd.DataFrame(pca.transform(Xvalid_data_frame))
Xtest_data_frame_pca = pd.DataFrame(pca.transform(Xtest_data_frame))
print('Shape after PCA: ',Xtrain_data_frame_pca.shape)

# %% [markdown]
# # SVM

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T10:12:55.404043Z","iopub.execute_input":"2023-09-24T10:12:55.404823Z","iopub.status.idle":"2023-09-24T10:13:35.526883Z","shell.execute_reply.started":"2023-09-24T10:12:55.404780Z","shell.execute_reply":"2023-09-24T10:13:35.525682Z"}}
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

classifier = svm.SVC(kernel='linear', C=1)

classifier.fit(Xtrain_data_frame_pca, ytrain_data_frame)

yvalid_prediction = classifier.predict(Xvalid_data_frame_pca)

print("acc_score: ",metrics.accuracy_score(yvalid_data_frame, yvalid_prediction))

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T10:13:35.528693Z","iopub.execute_input":"2023-09-24T10:13:35.529053Z","iopub.status.idle":"2023-09-24T10:55:39.910099Z","shell.execute_reply.started":"2023-09-24T10:13:35.529015Z","shell.execute_reply":"2023-09-24T10:55:39.908054Z"}}
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
import numpy as np

param_dist = {
    'C': [100,10,1,0,0.1,0.01],
    'kernel': ['rbf','linear','poly','sigmoid'],
    'gamma': ['scale','auto'],
    'degree': [1,2,3,4],
    'class_weight' : ['none','balanced']
}

svm = SVC()

random_search = RandomizedSearchCV(
    svm, param_distributions=param_dist, n_iter=10, cv=5, n_jobs=-1, random_state=42, scoring='accuracy'
)

random_search.fit(Xtrain_data_frame_pca, ytrain_data_frame)

best_params = random_search.best_params_
best_model = random_search.best_estimator_

print("best parameters:", best_params)

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T10:59:40.816500Z","iopub.execute_input":"2023-09-24T10:59:40.817027Z","iopub.status.idle":"2023-09-24T11:01:42.455221Z","shell.execute_reply.started":"2023-09-24T10:59:40.816989Z","shell.execute_reply":"2023-09-24T11:01:42.454200Z"}}
from sklearn import svm

classifier = svm.SVC(kernel='rbf', C=100, gamma='scale', degree=4, class_weight='balanced')

classifier.fit(Xtrain_data_frame_pca, ytrain_data_frame)

yvalid_prediction = classifier.predict(Xvalid_data_frame_pca)

print("acc_score: ",metrics.accuracy_score(yvalid_data_frame, yvalid_prediction))

ytest_predicticon_after_pca = classifier.predict(Xtest_data_frame_pca)

# %% [markdown]
# # **Random Forest**

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T11:13:20.033410Z","iopub.execute_input":"2023-09-24T11:13:20.034164Z","iopub.status.idle":"2023-09-24T11:17:10.791320Z","shell.execute_reply.started":"2023-09-24T11:13:20.034130Z","shell.execute_reply":"2023-09-24T11:17:10.790224Z"}}
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=100, random_state=42)

classifier.fit(Xtrain_data_frame, ytrain_data_frame)

yvalid_prediction = classifier.predict(Xvalid_data_frame)

print("accuracy_score: ",metrics.accuracy_score(yvalid_data_frame, yvalid_prediction))

y_test_pred = classifier.predict(Xtest_data_frame)

# %% [markdown]
# # CSV Creation

# %% [code]
output_data_frame=pd.DataFrame(columns=["ID","label_1","label_2","label_3","label_4"])

# %% [code]
IDs = list(i for i in range(1, len(test_df)+1))
output_data_frame["ID"] = IDs

# %% [code]
output_data_frame["label_4"] = ytest_prediction_after_pca

# %% [code]
output_data_frame

# %% [code]
output_df.to_csv('/kaggle/working/output.csv',index=False)

# %% [code]
