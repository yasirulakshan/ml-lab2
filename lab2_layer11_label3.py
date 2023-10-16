{"metadata":{"kernelspec":{"language":"python","display_name":"Python 3","name":"python3"},"language_info":{"name":"python","version":"3.10.12","mimetype":"text/x-python","codemirror_mode":{"name":"ipython","version":3},"pygments_lexer":"ipython3","nbconvert_exporter":"python","file_extension":".py"}},"nbformat_minor":4,"nbformat":4,"cells":[{"cell_type":"code","source":"import pandas as pd\nimport numpy as np\nfrom pandas import Series\n\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.ensemble import RandomForestClassifier\nfrom sklearn.decomposition import PCA\nfrom sklearn.metrics import f1_score\nimport seaborn as sns\n\nimport matplotlib.pyplot as plt\n%matplotlib inline\n\nimport warnings\nwarnings.filterwarnings(\"ignore\")","metadata":{"_uuid":"8f2839f25d086af736a60e9eeb907d3b93b6e0e5","_cell_guid":"b1076dfc-b9ad-4769-8c92-a6c4dae69d19","execution":{"iopub.status.busy":"2023-09-24T10:42:39.779724Z","iopub.execute_input":"2023-09-24T10:42:39.780884Z","iopub.status.idle":"2023-09-24T10:42:39.790296Z","shell.execute_reply.started":"2023-09-24T10:42:39.780812Z","shell.execute_reply":"2023-09-24T10:42:39.789148Z"},"trusted":true},"execution_count":9,"outputs":[]},{"cell_type":"code","source":"train_data_frame = pd.read_csv('/kaggle/input/layer11/train.csv')\nvalid_data_frame = pd.read_csv('/kaggle/input/layer11/valid.csv')\ntest_data_frame = pd.read_csv('/kaggle/input/layer11/test.csv')","metadata":{"execution":{"iopub.status.busy":"2023-09-24T10:42:45.008008Z","iopub.execute_input":"2023-09-24T10:42:45.008433Z","iopub.status.idle":"2023-09-24T10:42:52.448658Z","shell.execute_reply.started":"2023-09-24T10:42:45.008399Z","shell.execute_reply":"2023-09-24T10:42:52.447379Z"},"trusted":true},"execution_count":10,"outputs":[]},{"cell_type":"code","source":"train_data_frame.shape","metadata":{"execution":{"iopub.status.busy":"2023-09-24T10:42:52.450878Z","iopub.execute_input":"2023-09-24T10:42:52.451327Z","iopub.status.idle":"2023-09-24T10:42:52.459437Z","shell.execute_reply.started":"2023-09-24T10:42:52.451294Z","shell.execute_reply":"2023-09-24T10:42:52.458227Z"},"trusted":true},"execution_count":11,"outputs":[{"execution_count":11,"output_type":"execute_result","data":{"text/plain":"(28520, 772)"},"metadata":{}}]},{"cell_type":"code","source":"train_data_frame.head()","metadata":{"execution":{"iopub.status.busy":"2023-09-24T10:42:55.627035Z","iopub.execute_input":"2023-09-24T10:42:55.627434Z","iopub.status.idle":"2023-09-24T10:42:55.660042Z","shell.execute_reply.started":"2023-09-24T10:42:55.627403Z","shell.execute_reply":"2023-09-24T10:42:55.658972Z"},"trusted":true},"execution_count":12,"outputs":[{"execution_count":12,"output_type":"execute_result","data":{"text/plain":"   feature_1  feature_2  feature_3  feature_4  feature_5  feature_6  \\\n0   0.031138   0.079892   0.157382  -0.014636  -0.051778  -0.021332   \n1   0.113040   0.175731   0.217741  -0.196254  -0.010129  -0.030586   \n2   0.048570   0.091281   0.160776  -0.150937   0.020115   0.044117   \n3   0.039212   0.118388   0.173831  -0.096659  -0.008702   0.061298   \n4   0.056019   0.170639   0.157917  -0.228605  -0.065965  -0.088732   \n\n   feature_7  feature_8  feature_9  feature_10  ...  feature_763  feature_764  \\\n0  -0.073593  -0.005386  -0.212557    0.099683  ...    -0.085248    -0.096007   \n1   0.067114  -0.072412  -0.239192    0.104741  ...    -0.090283    -0.053885   \n2  -0.050092  -0.045661  -0.155332    0.117206  ...    -0.021524    -0.008411   \n3   0.008974  -0.003277  -0.065046    0.095480  ...    -0.071936    -0.023120   \n4  -0.082243  -0.080568  -0.341500    0.142430  ...    -0.155621    -0.079447   \n\n   feature_765  feature_766  feature_767  feature_768  label_1  label_2  \\\n0    -0.000766     0.021399    -0.041432     0.094806       45      NaN   \n1    -0.010967     0.062209    -0.122958     0.192949       45      NaN   \n2    -0.006248     0.031468    -0.056915     0.154731       45      NaN   \n3    -0.007812     0.057600    -0.121892     0.072796       45      NaN   \n4     0.015316     0.127726    -0.151966     0.169634       45      NaN   \n\n   label_3  label_4  \n0        1        6  \n1        1        6  \n2        1        6  \n3        1        6  \n4        1        6  \n\n[5 rows x 772 columns]","text/html":"<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>feature_1</th>\n      <th>feature_2</th>\n      <th>feature_3</th>\n      <th>feature_4</th>\n      <th>feature_5</th>\n      <th>feature_6</th>\n      <th>feature_7</th>\n      <th>feature_8</th>\n      <th>feature_9</th>\n      <th>feature_10</th>\n      <th>...</th>\n      <th>feature_763</th>\n      <th>feature_764</th>\n      <th>feature_765</th>\n      <th>feature_766</th>\n      <th>feature_767</th>\n      <th>feature_768</th>\n      <th>label_1</th>\n      <th>label_2</th>\n      <th>label_3</th>\n      <th>label_4</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>0</th>\n      <td>0.031138</td>\n      <td>0.079892</td>\n      <td>0.157382</td>\n      <td>-0.014636</td>\n      <td>-0.051778</td>\n      <td>-0.021332</td>\n      <td>-0.073593</td>\n      <td>-0.005386</td>\n      <td>-0.212557</td>\n      <td>0.099683</td>\n      <td>...</td>\n      <td>-0.085248</td>\n      <td>-0.096007</td>\n      <td>-0.000766</td>\n      <td>0.021399</td>\n      <td>-0.041432</td>\n      <td>0.094806</td>\n      <td>45</td>\n      <td>NaN</td>\n      <td>1</td>\n      <td>6</td>\n    </tr>\n    <tr>\n      <th>1</th>\n      <td>0.113040</td>\n      <td>0.175731</td>\n      <td>0.217741</td>\n      <td>-0.196254</td>\n      <td>-0.010129</td>\n      <td>-0.030586</td>\n      <td>0.067114</td>\n      <td>-0.072412</td>\n      <td>-0.239192</td>\n      <td>0.104741</td>\n      <td>...</td>\n      <td>-0.090283</td>\n      <td>-0.053885</td>\n      <td>-0.010967</td>\n      <td>0.062209</td>\n      <td>-0.122958</td>\n      <td>0.192949</td>\n      <td>45</td>\n      <td>NaN</td>\n      <td>1</td>\n      <td>6</td>\n    </tr>\n    <tr>\n      <th>2</th>\n      <td>0.048570</td>\n      <td>0.091281</td>\n      <td>0.160776</td>\n      <td>-0.150937</td>\n      <td>0.020115</td>\n      <td>0.044117</td>\n      <td>-0.050092</td>\n      <td>-0.045661</td>\n      <td>-0.155332</td>\n      <td>0.117206</td>\n      <td>...</td>\n      <td>-0.021524</td>\n      <td>-0.008411</td>\n      <td>-0.006248</td>\n      <td>0.031468</td>\n      <td>-0.056915</td>\n      <td>0.154731</td>\n      <td>45</td>\n      <td>NaN</td>\n      <td>1</td>\n      <td>6</td>\n    </tr>\n    <tr>\n      <th>3</th>\n      <td>0.039212</td>\n      <td>0.118388</td>\n      <td>0.173831</td>\n      <td>-0.096659</td>\n      <td>-0.008702</td>\n      <td>0.061298</td>\n      <td>0.008974</td>\n      <td>-0.003277</td>\n      <td>-0.065046</td>\n      <td>0.095480</td>\n      <td>...</td>\n      <td>-0.071936</td>\n      <td>-0.023120</td>\n      <td>-0.007812</td>\n      <td>0.057600</td>\n      <td>-0.121892</td>\n      <td>0.072796</td>\n      <td>45</td>\n      <td>NaN</td>\n      <td>1</td>\n      <td>6</td>\n    </tr>\n    <tr>\n      <th>4</th>\n      <td>0.056019</td>\n      <td>0.170639</td>\n      <td>0.157917</td>\n      <td>-0.228605</td>\n      <td>-0.065965</td>\n      <td>-0.088732</td>\n      <td>-0.082243</td>\n      <td>-0.080568</td>\n      <td>-0.341500</td>\n      <td>0.142430</td>\n      <td>...</td>\n      <td>-0.155621</td>\n      <td>-0.079447</td>\n      <td>0.015316</td>\n      <td>0.127726</td>\n      <td>-0.151966</td>\n      <td>0.169634</td>\n      <td>45</td>\n      <td>NaN</td>\n      <td>1</td>\n      <td>6</td>\n    </tr>\n  </tbody>\n</table>\n<p>5 rows × 772 columns</p>\n</div>"},"metadata":{}}]},{"cell_type":"code","source":"test_data_frame.head()","metadata":{"execution":{"iopub.status.busy":"2023-09-24T10:42:58.796536Z","iopub.execute_input":"2023-09-24T10:42:58.796939Z","iopub.status.idle":"2023-09-24T10:42:58.825169Z","shell.execute_reply.started":"2023-09-24T10:42:58.796905Z","shell.execute_reply":"2023-09-24T10:42:58.824149Z"},"trusted":true},"execution_count":13,"outputs":[{"execution_count":13,"output_type":"execute_result","data":{"text/plain":"   ID  feature_1  feature_2  feature_3  feature_4  feature_5  feature_6  \\\n0   1   0.124623   0.196628   0.257004  -0.156045  -0.054916   0.006071   \n1   2   0.109655   0.170158   0.227644  -0.127088  -0.044476  -0.046852   \n2   3   0.014854   0.030051   0.115092  -0.017179   0.002720  -0.011692   \n3   4   0.196893   0.113314   0.352175  -0.108499  -0.064472  -0.073239   \n4   5   0.033004   0.013373   0.124001  -0.016143   0.010120   0.010635   \n\n   feature_7  feature_8  feature_9  ...  feature_759  feature_760  \\\n0  -0.035149  -0.092019  -0.196302  ...    -0.221466     0.140292   \n1  -0.090026  -0.061321  -0.227288  ...    -0.204930     0.110203   \n2  -0.078855  -0.042991  -0.096283  ...    -0.032937     0.075821   \n3  -0.086402   0.008671  -0.342217  ...    -0.255167     0.096579   \n4  -0.055789  -0.036282  -0.059422  ...    -0.035814     0.093764   \n\n   feature_761  feature_762  feature_763  feature_764  feature_765  \\\n0     0.123622    -0.175572    -0.107030    -0.087621    -0.026501   \n1     0.085665    -0.286787    -0.113195    -0.057312    -0.055680   \n2     0.030987    -0.149850    -0.003155    -0.010207    -0.001427   \n3     0.069413    -0.215386    -0.075168    -0.035071    -0.023375   \n4     0.027321    -0.116009     0.010096    -0.042293     0.005347   \n\n   feature_766  feature_767  feature_768  \n0     0.139337    -0.083030     0.059507  \n1     0.143939    -0.045760     0.106113  \n2     0.000934    -0.017069     0.048123  \n3     0.067768    -0.181530     0.174444  \n4     0.007722    -0.007731     0.058799  \n\n[5 rows x 769 columns]","text/html":"<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>ID</th>\n      <th>feature_1</th>\n      <th>feature_2</th>\n      <th>feature_3</th>\n      <th>feature_4</th>\n      <th>feature_5</th>\n      <th>feature_6</th>\n      <th>feature_7</th>\n      <th>feature_8</th>\n      <th>feature_9</th>\n      <th>...</th>\n      <th>feature_759</th>\n      <th>feature_760</th>\n      <th>feature_761</th>\n      <th>feature_762</th>\n      <th>feature_763</th>\n      <th>feature_764</th>\n      <th>feature_765</th>\n      <th>feature_766</th>\n      <th>feature_767</th>\n      <th>feature_768</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>0</th>\n      <td>1</td>\n      <td>0.124623</td>\n      <td>0.196628</td>\n      <td>0.257004</td>\n      <td>-0.156045</td>\n      <td>-0.054916</td>\n      <td>0.006071</td>\n      <td>-0.035149</td>\n      <td>-0.092019</td>\n      <td>-0.196302</td>\n      <td>...</td>\n      <td>-0.221466</td>\n      <td>0.140292</td>\n      <td>0.123622</td>\n      <td>-0.175572</td>\n      <td>-0.107030</td>\n      <td>-0.087621</td>\n      <td>-0.026501</td>\n      <td>0.139337</td>\n      <td>-0.083030</td>\n      <td>0.059507</td>\n    </tr>\n    <tr>\n      <th>1</th>\n      <td>2</td>\n      <td>0.109655</td>\n      <td>0.170158</td>\n      <td>0.227644</td>\n      <td>-0.127088</td>\n      <td>-0.044476</td>\n      <td>-0.046852</td>\n      <td>-0.090026</td>\n      <td>-0.061321</td>\n      <td>-0.227288</td>\n      <td>...</td>\n      <td>-0.204930</td>\n      <td>0.110203</td>\n      <td>0.085665</td>\n      <td>-0.286787</td>\n      <td>-0.113195</td>\n      <td>-0.057312</td>\n      <td>-0.055680</td>\n      <td>0.143939</td>\n      <td>-0.045760</td>\n      <td>0.106113</td>\n    </tr>\n    <tr>\n      <th>2</th>\n      <td>3</td>\n      <td>0.014854</td>\n      <td>0.030051</td>\n      <td>0.115092</td>\n      <td>-0.017179</td>\n      <td>0.002720</td>\n      <td>-0.011692</td>\n      <td>-0.078855</td>\n      <td>-0.042991</td>\n      <td>-0.096283</td>\n      <td>...</td>\n      <td>-0.032937</td>\n      <td>0.075821</td>\n      <td>0.030987</td>\n      <td>-0.149850</td>\n      <td>-0.003155</td>\n      <td>-0.010207</td>\n      <td>-0.001427</td>\n      <td>0.000934</td>\n      <td>-0.017069</td>\n      <td>0.048123</td>\n    </tr>\n    <tr>\n      <th>3</th>\n      <td>4</td>\n      <td>0.196893</td>\n      <td>0.113314</td>\n      <td>0.352175</td>\n      <td>-0.108499</td>\n      <td>-0.064472</td>\n      <td>-0.073239</td>\n      <td>-0.086402</td>\n      <td>0.008671</td>\n      <td>-0.342217</td>\n      <td>...</td>\n      <td>-0.255167</td>\n      <td>0.096579</td>\n      <td>0.069413</td>\n      <td>-0.215386</td>\n      <td>-0.075168</td>\n      <td>-0.035071</td>\n      <td>-0.023375</td>\n      <td>0.067768</td>\n      <td>-0.181530</td>\n      <td>0.174444</td>\n    </tr>\n    <tr>\n      <th>4</th>\n      <td>5</td>\n      <td>0.033004</td>\n      <td>0.013373</td>\n      <td>0.124001</td>\n      <td>-0.016143</td>\n      <td>0.010120</td>\n      <td>0.010635</td>\n      <td>-0.055789</td>\n      <td>-0.036282</td>\n      <td>-0.059422</td>\n      <td>...</td>\n      <td>-0.035814</td>\n      <td>0.093764</td>\n      <td>0.027321</td>\n      <td>-0.116009</td>\n      <td>0.010096</td>\n      <td>-0.042293</td>\n      <td>0.005347</td>\n      <td>0.007722</td>\n      <td>-0.007731</td>\n      <td>0.058799</td>\n    </tr>\n  </tbody>\n</table>\n<p>5 rows × 769 columns</p>\n</div>"},"metadata":{}}]},{"cell_type":"code","source":"missing_cols = train_data_frame.columns[train_data_frame.isnull().any()]\nmissing_counts = train_data_frame[missing_cols].isnull().sum()\n\nprint('Missing Columns and Counts')\nfor column in missing_cols:\n    print( str(column) +' : '+ str(missing_counts[column]))","metadata":{"execution":{"iopub.status.busy":"2023-09-24T10:43:02.367646Z","iopub.execute_input":"2023-09-24T10:43:02.368045Z","iopub.status.idle":"2023-09-24T10:43:02.399494Z","shell.execute_reply.started":"2023-09-24T10:43:02.368011Z","shell.execute_reply":"2023-09-24T10:43:02.398191Z"},"trusted":true},"execution_count":14,"outputs":[{"name":"stdout","text":"Missing Columns and Counts\nlabel_2 : 480\n","output_type":"stream"}]},{"cell_type":"code","source":"train_data = train_data_frame.copy()\nvalid_data = valid_data_frame.copy()\ntest_data = test_data_frame.copy()","metadata":{"execution":{"iopub.status.busy":"2023-09-24T10:43:05.274623Z","iopub.execute_input":"2023-09-24T10:43:05.275172Z","iopub.status.idle":"2023-09-24T10:43:05.353948Z","shell.execute_reply.started":"2023-09-24T10:43:05.275124Z","shell.execute_reply":"2023-09-24T10:43:05.352609Z"},"trusted":true},"execution_count":15,"outputs":[]},{"cell_type":"code","source":"train_data_frame.describe()","metadata":{"execution":{"iopub.status.busy":"2023-09-24T10:43:09.083063Z","iopub.execute_input":"2023-09-24T10:43:09.083472Z","iopub.status.idle":"2023-09-24T10:43:11.711122Z","shell.execute_reply.started":"2023-09-24T10:43:09.083440Z","shell.execute_reply":"2023-09-24T10:43:11.709870Z"},"trusted":true},"execution_count":16,"outputs":[{"execution_count":16,"output_type":"execute_result","data":{"text/plain":"          feature_1     feature_2     feature_3     feature_4     feature_5  \\\ncount  28520.000000  28520.000000  28520.000000  28520.000000  28520.000000   \nmean       0.042487      0.068749      0.145547     -0.070646     -0.013539   \nstd        0.048918      0.046354      0.065332      0.046671      0.027635   \nmin       -0.079594     -0.062608     -0.048545     -0.307243     -0.178347   \n25%        0.009979      0.037225      0.100677     -0.088834     -0.027810   \n50%        0.024445      0.056119      0.123554     -0.057386     -0.011423   \n75%        0.058410      0.086358      0.173234     -0.039661      0.001544   \nmax        0.274146      0.332288      0.454182      0.059362      0.196950   \n\n          feature_6     feature_7     feature_8     feature_9    feature_10  \\\ncount  28520.000000  28520.000000  28520.000000  28520.000000  28520.000000   \nmean       0.003395     -0.041282     -0.028283     -0.106602      0.053686   \nstd        0.031248      0.026479      0.029632      0.070775      0.030945   \nmin       -0.194771     -0.197551     -0.304828     -0.421257     -0.049723   \n25%       -0.010617     -0.056682     -0.044344     -0.135110      0.033236   \n50%        0.006173     -0.041501     -0.025805     -0.080715      0.045567   \n75%        0.021250     -0.026559     -0.009324     -0.057873      0.065670   \nmax        0.213127      0.124194      0.105714      0.192121      0.252320   \n\n       ...   feature_763   feature_764   feature_765   feature_766  \\\ncount  ...  28520.000000  28520.000000  28520.000000  28520.000000   \nmean   ...     -0.022102     -0.044743     -0.004380      0.049072   \nstd    ...      0.053250      0.031361      0.025829      0.050536   \nmin    ...     -0.253255     -0.264549     -0.137827     -0.117697   \n25%    ...     -0.042332     -0.056918     -0.018848      0.012599   \n50%    ...     -0.007960     -0.037407     -0.004701      0.033121   \n75%    ...      0.012116     -0.024179      0.010218      0.072599   \nmax    ...      0.209455      0.054555      0.215375      0.376414   \n\n        feature_767   feature_768       label_1       label_2       label_3  \\\ncount  28520.000000  28520.000000  28520.000000  28040.000000  28520.000000   \nmean      -0.028722      0.075717     30.498843     27.975107      0.799299   \nstd        0.032622      0.044879     17.328389      5.735913      0.400532   \nmin       -0.302399     -0.090777      1.000000     22.000000      0.000000   \n25%       -0.045226      0.045309     15.000000     25.000000      1.000000   \n50%       -0.022919      0.064875     30.000000     27.000000      1.000000   \n75%       -0.006335      0.097642     46.000000     30.000000      1.000000   \nmax        0.125857      0.416291     60.000000     61.000000      1.000000   \n\n            label_4  \ncount  28520.000000  \nmean       5.997125  \nstd        2.375567  \nmin        0.000000  \n25%        6.000000  \n50%        6.000000  \n75%        6.000000  \nmax       13.000000  \n\n[8 rows x 772 columns]","text/html":"<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>feature_1</th>\n      <th>feature_2</th>\n      <th>feature_3</th>\n      <th>feature_4</th>\n      <th>feature_5</th>\n      <th>feature_6</th>\n      <th>feature_7</th>\n      <th>feature_8</th>\n      <th>feature_9</th>\n      <th>feature_10</th>\n      <th>...</th>\n      <th>feature_763</th>\n      <th>feature_764</th>\n      <th>feature_765</th>\n      <th>feature_766</th>\n      <th>feature_767</th>\n      <th>feature_768</th>\n      <th>label_1</th>\n      <th>label_2</th>\n      <th>label_3</th>\n      <th>label_4</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>count</th>\n      <td>28520.000000</td>\n      <td>28520.000000</td>\n      <td>28520.000000</td>\n      <td>28520.000000</td>\n      <td>28520.000000</td>\n      <td>28520.000000</td>\n      <td>28520.000000</td>\n      <td>28520.000000</td>\n      <td>28520.000000</td>\n      <td>28520.000000</td>\n      <td>...</td>\n      <td>28520.000000</td>\n      <td>28520.000000</td>\n      <td>28520.000000</td>\n      <td>28520.000000</td>\n      <td>28520.000000</td>\n      <td>28520.000000</td>\n      <td>28520.000000</td>\n      <td>28040.000000</td>\n      <td>28520.000000</td>\n      <td>28520.000000</td>\n    </tr>\n    <tr>\n      <th>mean</th>\n      <td>0.042487</td>\n      <td>0.068749</td>\n      <td>0.145547</td>\n      <td>-0.070646</td>\n      <td>-0.013539</td>\n      <td>0.003395</td>\n      <td>-0.041282</td>\n      <td>-0.028283</td>\n      <td>-0.106602</td>\n      <td>0.053686</td>\n      <td>...</td>\n      <td>-0.022102</td>\n      <td>-0.044743</td>\n      <td>-0.004380</td>\n      <td>0.049072</td>\n      <td>-0.028722</td>\n      <td>0.075717</td>\n      <td>30.498843</td>\n      <td>27.975107</td>\n      <td>0.799299</td>\n      <td>5.997125</td>\n    </tr>\n    <tr>\n      <th>std</th>\n      <td>0.048918</td>\n      <td>0.046354</td>\n      <td>0.065332</td>\n      <td>0.046671</td>\n      <td>0.027635</td>\n      <td>0.031248</td>\n      <td>0.026479</td>\n      <td>0.029632</td>\n      <td>0.070775</td>\n      <td>0.030945</td>\n      <td>...</td>\n      <td>0.053250</td>\n      <td>0.031361</td>\n      <td>0.025829</td>\n      <td>0.050536</td>\n      <td>0.032622</td>\n      <td>0.044879</td>\n      <td>17.328389</td>\n      <td>5.735913</td>\n      <td>0.400532</td>\n      <td>2.375567</td>\n    </tr>\n    <tr>\n      <th>min</th>\n      <td>-0.079594</td>\n      <td>-0.062608</td>\n      <td>-0.048545</td>\n      <td>-0.307243</td>\n      <td>-0.178347</td>\n      <td>-0.194771</td>\n      <td>-0.197551</td>\n      <td>-0.304828</td>\n      <td>-0.421257</td>\n      <td>-0.049723</td>\n      <td>...</td>\n      <td>-0.253255</td>\n      <td>-0.264549</td>\n      <td>-0.137827</td>\n      <td>-0.117697</td>\n      <td>-0.302399</td>\n      <td>-0.090777</td>\n      <td>1.000000</td>\n      <td>22.000000</td>\n      <td>0.000000</td>\n      <td>0.000000</td>\n    </tr>\n    <tr>\n      <th>25%</th>\n      <td>0.009979</td>\n      <td>0.037225</td>\n      <td>0.100677</td>\n      <td>-0.088834</td>\n      <td>-0.027810</td>\n      <td>-0.010617</td>\n      <td>-0.056682</td>\n      <td>-0.044344</td>\n      <td>-0.135110</td>\n      <td>0.033236</td>\n      <td>...</td>\n      <td>-0.042332</td>\n      <td>-0.056918</td>\n      <td>-0.018848</td>\n      <td>0.012599</td>\n      <td>-0.045226</td>\n      <td>0.045309</td>\n      <td>15.000000</td>\n      <td>25.000000</td>\n      <td>1.000000</td>\n      <td>6.000000</td>\n    </tr>\n    <tr>\n      <th>50%</th>\n      <td>0.024445</td>\n      <td>0.056119</td>\n      <td>0.123554</td>\n      <td>-0.057386</td>\n      <td>-0.011423</td>\n      <td>0.006173</td>\n      <td>-0.041501</td>\n      <td>-0.025805</td>\n      <td>-0.080715</td>\n      <td>0.045567</td>\n      <td>...</td>\n      <td>-0.007960</td>\n      <td>-0.037407</td>\n      <td>-0.004701</td>\n      <td>0.033121</td>\n      <td>-0.022919</td>\n      <td>0.064875</td>\n      <td>30.000000</td>\n      <td>27.000000</td>\n      <td>1.000000</td>\n      <td>6.000000</td>\n    </tr>\n    <tr>\n      <th>75%</th>\n      <td>0.058410</td>\n      <td>0.086358</td>\n      <td>0.173234</td>\n      <td>-0.039661</td>\n      <td>0.001544</td>\n      <td>0.021250</td>\n      <td>-0.026559</td>\n      <td>-0.009324</td>\n      <td>-0.057873</td>\n      <td>0.065670</td>\n      <td>...</td>\n      <td>0.012116</td>\n      <td>-0.024179</td>\n      <td>0.010218</td>\n      <td>0.072599</td>\n      <td>-0.006335</td>\n      <td>0.097642</td>\n      <td>46.000000</td>\n      <td>30.000000</td>\n      <td>1.000000</td>\n      <td>6.000000</td>\n    </tr>\n    <tr>\n      <th>max</th>\n      <td>0.274146</td>\n      <td>0.332288</td>\n      <td>0.454182</td>\n      <td>0.059362</td>\n      <td>0.196950</td>\n      <td>0.213127</td>\n      <td>0.124194</td>\n      <td>0.105714</td>\n      <td>0.192121</td>\n      <td>0.252320</td>\n      <td>...</td>\n      <td>0.209455</td>\n      <td>0.054555</td>\n      <td>0.215375</td>\n      <td>0.376414</td>\n      <td>0.125857</td>\n      <td>0.416291</td>\n      <td>60.000000</td>\n      <td>61.000000</td>\n      <td>1.000000</td>\n      <td>13.000000</td>\n    </tr>\n  </tbody>\n</table>\n<p>8 rows × 772 columns</p>\n</div>"},"metadata":{}}]},{"cell_type":"code","source":"from sklearn.preprocessing import RobustScaler # eliminate outliers\n\nXtrain = {}\nytrain = {}\n\nXvalid = {}\nyvalid = {}\n\nXtest = {}\nytest = {}\n\n#create dictionaries for each label\nfor target_label in ['label_1','label_2','label_3','label_4']:\n\n  if target_label == \"label_2\":\n    train = train_data_frame[train_data_frame['label_2'].notna()]\n    valid = valid_data_frame[valid_data_frame['label_2'].notna()]\n  else:\n    train = train_data_frame\n    valid = valid_data_frame\n\n  test = test_data_frame\n\n  scaler = RobustScaler()\n\n  Xtrain[target_label] = pd.DataFrame(scaler.fit_transform(train.drop(['label_1','label_2','label_3','label_4'], axis=1)), columns=[f'feature_{i}' for i in range(1,769)])\n  ytrain[target_label] = train[target_label]\n\n  Xvalid[target_label] = pd.DataFrame(scaler.transform(valid.drop(['label_1','label_2','label_3','label_4'], axis=1)), columns=[f'feature_{i}' for i in range(1,769)])\n  yvalid  [target_label] = valid[target_label]\n\n  Xtest[target_label] = pd.DataFrame(scaler.transform(test.drop([\"ID\"],axis=1)), columns=[f'feature_{i}' for i in range(1,769)])","metadata":{"execution":{"iopub.status.busy":"2023-09-24T10:43:15.285795Z","iopub.execute_input":"2023-09-24T10:43:15.286210Z","iopub.status.idle":"2023-09-24T10:43:20.171328Z","shell.execute_reply.started":"2023-09-24T10:43:15.286178Z","shell.execute_reply":"2023-09-24T10:43:20.170122Z"},"trusted":true},"execution_count":17,"outputs":[]},{"cell_type":"code","source":"Xtrain_data_frame = Xtrain['label_3'].copy()\nytrain_data_frame = ytrain['label_3'].copy()\n\nXvalid_data_frame = Xvalid['label_3'].copy()\nyvalid_data_frame = yvalid['label_3'].copy()\n\nXtest_data_frame = Xtest['label_3'].copy()","metadata":{"execution":{"iopub.status.busy":"2023-09-24T10:43:20.173311Z","iopub.execute_input":"2023-09-24T10:43:20.173655Z","iopub.status.idle":"2023-09-24T10:43:20.249361Z","shell.execute_reply.started":"2023-09-24T10:43:20.173624Z","shell.execute_reply":"2023-09-24T10:43:20.248044Z"},"trusted":true},"execution_count":18,"outputs":[]},{"cell_type":"code","source":"from sklearn.svm import SVC\nfrom sklearn.model_selection import cross_val_score, KFold\n\n# Perform cross-validation\nscores = cross_val_score(SVC(), Xtrain_data_frame, ytrain_data_frame, cv=5, scoring='accuracy')\n\nmean_accuracy = scores.mean()\nstd_accuracy = scores.std()\n\n# Print the cross-validation scores\nprint('Support Vector Machines')\nprint('\\n')\nprint(\"Cross-validation scores:\", scores)\nprint(f\"Mean Accuracy: {mean_accuracy:.2f}\")\nprint(f\"Standard Deviation: {std_accuracy:.2f}\")","metadata":{"execution":{"iopub.status.busy":"2023-09-24T10:43:22.952155Z","iopub.execute_input":"2023-09-24T10:43:22.952536Z","iopub.status.idle":"2023-09-24T10:48:12.871892Z","shell.execute_reply.started":"2023-09-24T10:43:22.952506Z","shell.execute_reply":"2023-09-24T10:48:12.870625Z"},"trusted":true},"execution_count":19,"outputs":[{"name":"stdout","text":"Support Vector Machines\n\n\nCross-validation scores: [0.97335203 0.99403927 0.9865007  0.9651122  0.98579944]\nMean Accuracy: 0.98\nStandard Deviation: 0.01\n","output_type":"stream"}]},{"cell_type":"code","source":"from sklearn.decomposition import PCA\n\npca = PCA(n_components=0.975, svd_solver='full')\npca.fit(Xtrain_data_frame)\nXtrain_data_frame_pca = pd.DataFrame(pca.transform(Xtrain_data_frame))\nXvalid_data_frame_pca = pd.DataFrame(pca.transform(Xvalid_data_frame))\nXtest_data_frame_pca = pd.DataFrame(pca.transform(Xtest_data_frame))\nprint('Shape after PCA: ',Xtrain_data_frame_pca.shape)","metadata":{"execution":{"iopub.status.busy":"2023-09-24T10:48:12.874432Z","iopub.execute_input":"2023-09-24T10:48:12.874886Z","iopub.status.idle":"2023-09-24T10:48:17.964033Z","shell.execute_reply.started":"2023-09-24T10:48:12.874819Z","shell.execute_reply":"2023-09-24T10:48:17.962898Z"},"trusted":true},"execution_count":20,"outputs":[{"name":"stdout","text":"Shape after PCA:  (28520, 245)\n","output_type":"stream"}]},{"cell_type":"markdown","source":"# SVM","metadata":{}},{"cell_type":"code","source":"from sklearn import svm\nfrom sklearn import metrics\nfrom sklearn.metrics import f1_score\nfrom sklearn.metrics import confusion_matrix\nfrom sklearn.metrics import classification_report\n\nclassifier = svm.SVC(kernel='linear', C=1)\n\nclassifier.fit(Xtrain_data_frame_pca, ytrain_data_frame)\n\nyvalid_prediction = classifier.predict(Xvalid_data_frame_pca)\n\nprint(\"acc_score: \",metrics.accuracy_score(yvalid_data_frame, yvalid_prediction))","metadata":{"execution":{"iopub.status.busy":"2023-09-24T10:48:17.965488Z","iopub.execute_input":"2023-09-24T10:48:17.966247Z","iopub.status.idle":"2023-09-24T10:48:31.323390Z","shell.execute_reply.started":"2023-09-24T10:48:17.966203Z","shell.execute_reply":"2023-09-24T10:48:31.322245Z"},"trusted":true},"execution_count":21,"outputs":[{"name":"stdout","text":"acc_score:  0.992\n","output_type":"stream"}]},{"cell_type":"code","source":"from sklearn.svm import SVC\nfrom sklearn.model_selection import RandomizedSearchCV\nfrom scipy.stats import uniform, randint\nimport numpy as np\n\nparam_dist = {\n    'C': [100,10,1,0,0.1,0.01],\n    'kernel': ['rbf','linear','poly','sigmoid'],\n    'gamma': ['scale','auto'],\n    'degree': [1,2,3,4],\n    'class_weight' : ['none','balanced']\n}\n\nsvm = SVC()\n\nrandom_search = RandomizedSearchCV(\n    svm, param_distributions=param_dist, n_iter=10, cv=5, n_jobs=-1, random_state=42, scoring='accuracy'\n)\n\nrandom_search.fit(Xtrain_data_frame_pca, ytrain_data_frame)\n\nbest_params = random_search.best_params_\nbest_model = random_search.best_estimator_\n\nprint(\"best parameters:\", best_params)","metadata":{"execution":{"iopub.status.busy":"2023-09-24T11:28:24.712412Z","iopub.execute_input":"2023-09-24T11:28:24.712853Z","iopub.status.idle":"2023-09-24T11:55:28.134716Z","shell.execute_reply.started":"2023-09-24T11:28:24.712802Z","shell.execute_reply":"2023-09-24T11:55:28.133011Z"},"trusted":true},"execution_count":23,"outputs":[{"name":"stdout","text":"best parameters: {'kernel': 'rbf', 'gamma': 'scale', 'degree': 4, 'class_weight': 'balanced', 'C': 100}\n","output_type":"stream"}]},{"cell_type":"code","source":"from sklearn import svm\n\nclassifier = svm.SVC(kernel='rbf', C=100, gamma='scale', degree=4, class_weight='balanced')\n\nclassifier.fit(Xtrain_data_frame_pca, ytrain_data_frame)\n\nyvalid_prediction = classifier.predict(Xvalid_data_frame_pca)\n\nprint(\"acc_score: \",metrics.accuracy_score(yvalid_data_frame, yvalid_prediction))\n\nytest_predicticon_after_pca = classifier.predict(Xtest_data_frame_pca)","metadata":{"execution":{"iopub.status.busy":"2023-09-24T11:56:58.197934Z","iopub.execute_input":"2023-09-24T11:56:58.198355Z","iopub.status.idle":"2023-09-24T11:57:17.840859Z","shell.execute_reply.started":"2023-09-24T11:56:58.198319Z","shell.execute_reply":"2023-09-24T11:57:17.839659Z"},"trusted":true},"execution_count":24,"outputs":[{"name":"stdout","text":"acc_score:  0.996\n","output_type":"stream"}]},{"cell_type":"markdown","source":"# **Random Forest**","metadata":{}},{"cell_type":"code","source":"from sklearn.ensemble import RandomForestClassifier\n\nclassifier = RandomForestClassifier(n_estimators=100, random_state=42)\n\nclassifier.fit(Xtrain_data_frame, ytrain_data_frame)\n\nyvalid_prediction = classifier.predict(Xvalid_data_frame)\n\nprint(\"accuracy_score: \",metrics.accuracy_score(yvalid_data_frame, yvalid_prediction))\n\ny_test_pred = classifier.predict(Xtest_data_frame)","metadata":{"execution":{"iopub.status.busy":"2023-09-24T11:57:17.843252Z","iopub.execute_input":"2023-09-24T11:57:17.843703Z","iopub.status.idle":"2023-09-24T11:59:04.893247Z","shell.execute_reply.started":"2023-09-24T11:57:17.843660Z","shell.execute_reply":"2023-09-24T11:59:04.891736Z"},"trusted":true},"execution_count":25,"outputs":[{"name":"stdout","text":"accuracy_score:  0.964\n","output_type":"stream"}]},{"cell_type":"markdown","source":"# CSV Creation","metadata":{}},{"cell_type":"code","source":"output_data_frame=pd.DataFrame(columns=[\"ID\",\"label_1\",\"label_2\",\"label_3\",\"label_4\"])","metadata":{"execution":{"iopub.status.busy":"2023-09-24T11:59:04.895249Z","iopub.execute_input":"2023-09-24T11:59:04.895882Z","iopub.status.idle":"2023-09-24T11:59:04.904009Z","shell.execute_reply.started":"2023-09-24T11:59:04.895832Z","shell.execute_reply":"2023-09-24T11:59:04.902826Z"},"trusted":true},"execution_count":26,"outputs":[]},{"cell_type":"code","source":"IDs = list(i for i in range(1, len(test_data_frame)+1))\noutput_data_frame[\"ID\"] = IDs","metadata":{"execution":{"iopub.status.busy":"2023-09-24T11:59:04.906885Z","iopub.execute_input":"2023-09-24T11:59:04.907212Z","iopub.status.idle":"2023-09-24T11:59:04.920143Z","shell.execute_reply.started":"2023-09-24T11:59:04.907184Z","shell.execute_reply":"2023-09-24T11:59:04.919019Z"},"trusted":true},"execution_count":27,"outputs":[]},{"cell_type":"code","source":"output_data_frame[\"label_3\"] = ytest_predicticon_after_pca","metadata":{"execution":{"iopub.status.busy":"2023-09-24T11:59:04.922042Z","iopub.execute_input":"2023-09-24T11:59:04.922423Z","iopub.status.idle":"2023-09-24T11:59:04.933588Z","shell.execute_reply.started":"2023-09-24T11:59:04.922390Z","shell.execute_reply":"2023-09-24T11:59:04.932592Z"},"trusted":true},"execution_count":28,"outputs":[]},{"cell_type":"code","source":"output_data_frame","metadata":{"execution":{"iopub.status.busy":"2023-09-24T11:59:04.934907Z","iopub.execute_input":"2023-09-24T11:59:04.935546Z","iopub.status.idle":"2023-09-24T11:59:04.959490Z","shell.execute_reply.started":"2023-09-24T11:59:04.935515Z","shell.execute_reply":"2023-09-24T11:59:04.958195Z"},"trusted":true},"execution_count":29,"outputs":[{"execution_count":29,"output_type":"execute_result","data":{"text/plain":"      ID label_1 label_2  label_3 label_4\n0      1     NaN     NaN        0     NaN\n1      2     NaN     NaN        1     NaN\n2      3     NaN     NaN        1     NaN\n3      4     NaN     NaN        1     NaN\n4      5     NaN     NaN        0     NaN\n..   ...     ...     ...      ...     ...\n739  740     NaN     NaN        1     NaN\n740  741     NaN     NaN        1     NaN\n741  742     NaN     NaN        1     NaN\n742  743     NaN     NaN        1     NaN\n743  744     NaN     NaN        1     NaN\n\n[744 rows x 5 columns]","text/html":"<div>\n<style scoped>\n    .dataframe tbody tr th:only-of-type {\n        vertical-align: middle;\n    }\n\n    .dataframe tbody tr th {\n        vertical-align: top;\n    }\n\n    .dataframe thead th {\n        text-align: right;\n    }\n</style>\n<table border=\"1\" class=\"dataframe\">\n  <thead>\n    <tr style=\"text-align: right;\">\n      <th></th>\n      <th>ID</th>\n      <th>label_1</th>\n      <th>label_2</th>\n      <th>label_3</th>\n      <th>label_4</th>\n    </tr>\n  </thead>\n  <tbody>\n    <tr>\n      <th>0</th>\n      <td>1</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>0</td>\n      <td>NaN</td>\n    </tr>\n    <tr>\n      <th>1</th>\n      <td>2</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>1</td>\n      <td>NaN</td>\n    </tr>\n    <tr>\n      <th>2</th>\n      <td>3</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>1</td>\n      <td>NaN</td>\n    </tr>\n    <tr>\n      <th>3</th>\n      <td>4</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>1</td>\n      <td>NaN</td>\n    </tr>\n    <tr>\n      <th>4</th>\n      <td>5</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>0</td>\n      <td>NaN</td>\n    </tr>\n    <tr>\n      <th>...</th>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n      <td>...</td>\n    </tr>\n    <tr>\n      <th>739</th>\n      <td>740</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>1</td>\n      <td>NaN</td>\n    </tr>\n    <tr>\n      <th>740</th>\n      <td>741</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>1</td>\n      <td>NaN</td>\n    </tr>\n    <tr>\n      <th>741</th>\n      <td>742</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>1</td>\n      <td>NaN</td>\n    </tr>\n    <tr>\n      <th>742</th>\n      <td>743</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>1</td>\n      <td>NaN</td>\n    </tr>\n    <tr>\n      <th>743</th>\n      <td>744</td>\n      <td>NaN</td>\n      <td>NaN</td>\n      <td>1</td>\n      <td>NaN</td>\n    </tr>\n  </tbody>\n</table>\n<p>744 rows × 5 columns</p>\n</div>"},"metadata":{}}]},{"cell_type":"code","source":"output_data_frame.to_csv('/kaggle/working/output.csv',index=False)","metadata":{"execution":{"iopub.status.busy":"2023-09-24T11:59:04.960974Z","iopub.execute_input":"2023-09-24T11:59:04.961883Z","iopub.status.idle":"2023-09-24T11:59:04.976160Z","shell.execute_reply.started":"2023-09-24T11:59:04.961825Z","shell.execute_reply":"2023-09-24T11:59:04.974905Z"},"trusted":true},"execution_count":30,"outputs":[]},{"cell_type":"code","source":"","metadata":{},"execution_count":null,"outputs":[]}]}