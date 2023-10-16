# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T08:39:13.204317Z","iopub.execute_input":"2023-09-24T08:39:13.204794Z","iopub.status.idle":"2023-09-24T08:39:13.214265Z","shell.execute_reply.started":"2023-09-24T08:39:13.204760Z","shell.execute_reply":"2023-09-24T08:39:13.213069Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T08:39:14.829856Z","iopub.execute_input":"2023-09-24T08:39:14.830323Z","iopub.status.idle":"2023-09-24T08:39:22.856018Z","shell.execute_reply.started":"2023-09-24T08:39:14.830288Z","shell.execute_reply":"2023-09-24T08:39:22.854636Z"}}
train_data_frame = pd.read_csv('/kaggle/input/layer11/train.csv')
valid_data_frame = pd.read_csv('/kaggle/input/layer11/valid.csv')
test_data_frame = pd.read_csv('/kaggle/input/layer11/test.csv')

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T08:39:22.859181Z","iopub.execute_input":"2023-09-24T08:39:22.859667Z","iopub.status.idle":"2023-09-24T08:39:22.869310Z","shell.execute_reply.started":"2023-09-24T08:39:22.859625Z","shell.execute_reply":"2023-09-24T08:39:22.867631Z"}}
train_data_frame.shape

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T08:39:22.871076Z","iopub.execute_input":"2023-09-24T08:39:22.871537Z","iopub.status.idle":"2023-09-24T08:39:22.904698Z","shell.execute_reply.started":"2023-09-24T08:39:22.871497Z","shell.execute_reply":"2023-09-24T08:39:22.902994Z"}}
train_data_frame.head()

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T08:39:22.908126Z","iopub.execute_input":"2023-09-24T08:39:22.909251Z","iopub.status.idle":"2023-09-24T08:39:22.948146Z","shell.execute_reply.started":"2023-09-24T08:39:22.909203Z","shell.execute_reply":"2023-09-24T08:39:22.947136Z"}}
test_data_frame.head()

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T08:39:27.074634Z","iopub.execute_input":"2023-09-24T08:39:27.075063Z","iopub.status.idle":"2023-09-24T08:39:27.117919Z","shell.execute_reply.started":"2023-09-24T08:39:27.075030Z","shell.execute_reply":"2023-09-24T08:39:27.116781Z"}}
missing_cols = train_data_frame.columns[train_data_frame.isnull().any()]
missing_counts = train_data_frame[missing_cols].isnull().sum()

print('Missing Columns and Counts')
for column in missing_cols:
    print( str(column) +' : '+ str(missing_counts[column]))

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T08:37:28.632017Z","iopub.execute_input":"2023-09-24T08:37:28.632488Z","iopub.status.idle":"2023-09-24T08:37:28.785489Z","shell.execute_reply.started":"2023-09-24T08:37:28.632455Z","shell.execute_reply":"2023-09-24T08:37:28.784213Z"}}
train_data = train_data_frame.copy()
valid_data = valid_data_frame.copy()
test_data = test_data_frame.copy()

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T08:39:30.357484Z","iopub.execute_input":"2023-09-24T08:39:30.357876Z","iopub.status.idle":"2023-09-24T08:39:33.548771Z","shell.execute_reply.started":"2023-09-24T08:39:30.357847Z","shell.execute_reply":"2023-09-24T08:39:33.547356Z"}}
train_data_frame.describe()

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T08:39:33.551472Z","iopub.execute_input":"2023-09-24T08:39:33.551848Z","iopub.status.idle":"2023-09-24T08:39:38.945039Z","shell.execute_reply.started":"2023-09-24T08:39:33.551818Z","shell.execute_reply":"2023-09-24T08:39:38.943974Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T08:39:38.947571Z","iopub.execute_input":"2023-09-24T08:39:38.948454Z","iopub.status.idle":"2023-09-24T08:39:39.041371Z","shell.execute_reply.started":"2023-09-24T08:39:38.948373Z","shell.execute_reply":"2023-09-24T08:39:39.039644Z"}}
Xtrain_data_frame = Xtrain['label_1'].copy()
ytrain_data_frame = ytrain['label_1'].copy()

Xvalid_data_frame = Xvalid['label_1'].copy()
yvalid_data_frame = yvalid['label_1'].copy()

Xtest_data_frame = Xtest['label_1'].copy()

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T08:39:39.043124Z","iopub.execute_input":"2023-09-24T08:39:39.044009Z","iopub.status.idle":"2023-09-24T09:01:28.377530Z","shell.execute_reply.started":"2023-09-24T08:39:39.043966Z","shell.execute_reply":"2023-09-24T09:01:28.375925Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T09:01:28.380160Z","iopub.execute_input":"2023-09-24T09:01:28.380610Z","iopub.status.idle":"2023-09-24T09:01:33.081812Z","shell.execute_reply.started":"2023-09-24T09:01:28.380576Z","shell.execute_reply":"2023-09-24T09:01:33.080579Z"}}
from sklearn.decomposition import PCA

pca = PCA(n_components=0.975, svd_solver='full')
pca.fit(Xtrain_data_frame)
Xtrain_data_frame_pca = pd.DataFrame(pca.transform(Xtrain_data_frame))
Xvalid_data_frame_pca = pd.DataFrame(pca.transform(Xvalid_data_frame))
Xtest_data_frame_pca = pd.DataFrame(pca.transform(Xtest_data_frame))
print('Shape after PCA: ',Xtrain_data_frame_pca.shape)

# %% [markdown]
# # SVM

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T09:01:33.083497Z","iopub.execute_input":"2023-09-24T09:01:33.084589Z","iopub.status.idle":"2023-09-24T09:02:07.293100Z","shell.execute_reply.started":"2023-09-24T09:01:33.084535Z","shell.execute_reply":"2023-09-24T09:02:07.291625Z"}}
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

classifier = svm.SVC(kernel='linear', C=1)

classifier.fit(Xtrain_data_frame_pca, ytrain_data_frame)

yvalid_prediction = classifier.predict(Xvalid_data_frame_pca)

print("acc_score: ",metrics.accuracy_score(yvalid_data_frame, yvalid_prediction))

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T09:58:10.862098Z","iopub.execute_input":"2023-09-24T09:58:10.862826Z","iopub.status.idle":"2023-09-24T10:25:27.894857Z","shell.execute_reply.started":"2023-09-24T09:58:10.862794Z","shell.execute_reply":"2023-09-24T10:25:27.893332Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T10:29:20.684284Z","iopub.execute_input":"2023-09-24T10:29:20.684748Z","iopub.status.idle":"2023-09-24T10:30:32.346668Z","shell.execute_reply.started":"2023-09-24T10:29:20.684715Z","shell.execute_reply":"2023-09-24T10:30:32.345439Z"}}
from sklearn import svm

classifier = svm.SVC(kernel='rbf', C=100, gamma='scale', degree=4, class_weight='balanced')

classifier.fit(Xtrain_data_frame_pca, ytrain_data_frame)

yvalid_prediction = classifier.predict(Xvalid_data_frame_pca)

print("acc_score: ",metrics.accuracy_score(yvalid_data_frame, yvalid_prediction))

ytest_predicticon_after_pca = classifier.predict(Xtest_data_frame_pca)

# %% [markdown]
# # **Random Forest**

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T10:30:32.349652Z","iopub.execute_input":"2023-09-24T10:30:32.350138Z","iopub.status.idle":"2023-09-24T10:34:16.412219Z","shell.execute_reply.started":"2023-09-24T10:30:32.350097Z","shell.execute_reply":"2023-09-24T10:34:16.410845Z"}}
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=100, random_state=42)

classifier.fit(Xtrain_data_frame, ytrain_data_frame)

yvalid_prediction = classifier.predict(Xvalid_data_frame)

print("accuracy_score: ",metrics.accuracy_score(yvalid_data_frame, yvalid_prediction))

y_test_pred = classifier.predict(Xtest_data_frame)

# %% [markdown]
# # CSV Creation

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T10:34:16.414242Z","iopub.execute_input":"2023-09-24T10:34:16.414697Z","iopub.status.idle":"2023-09-24T10:34:16.425907Z","shell.execute_reply.started":"2023-09-24T10:34:16.414664Z","shell.execute_reply":"2023-09-24T10:34:16.424187Z"}}
output_data_frame=pd.DataFrame(columns=["ID","label_1","label_2","label_3","label_4"])

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T10:34:31.105015Z","iopub.execute_input":"2023-09-24T10:34:31.105494Z","iopub.status.idle":"2023-09-24T10:34:31.117206Z","shell.execute_reply.started":"2023-09-24T10:34:31.105462Z","shell.execute_reply":"2023-09-24T10:34:31.115519Z"}}
IDs = list(i for i in range(1, len(test_data_frame)+1))
output_data_frame["ID"] = IDs

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T10:34:33.990546Z","iopub.execute_input":"2023-09-24T10:34:33.991708Z","iopub.status.idle":"2023-09-24T10:34:33.997813Z","shell.execute_reply.started":"2023-09-24T10:34:33.991666Z","shell.execute_reply":"2023-09-24T10:34:33.996491Z"}}
output_data_frame["label_1"] = ytest_predicticon_after_pca

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T10:34:37.495588Z","iopub.execute_input":"2023-09-24T10:34:37.495988Z","iopub.status.idle":"2023-09-24T10:34:37.516822Z","shell.execute_reply.started":"2023-09-24T10:34:37.495959Z","shell.execute_reply":"2023-09-24T10:34:37.515646Z"}}
output_data_frame

# %% [code] {"execution":{"iopub.status.busy":"2023-09-24T10:34:40.993296Z","iopub.execute_input":"2023-09-24T10:34:40.993777Z","iopub.status.idle":"2023-09-24T10:34:41.008164Z","shell.execute_reply.started":"2023-09-24T10:34:40.993746Z","shell.execute_reply":"2023-09-24T10:34:41.006317Z"}}
output_data_frame.to_csv('/kaggle/working/output.csv',index=False)

# %% [code]
