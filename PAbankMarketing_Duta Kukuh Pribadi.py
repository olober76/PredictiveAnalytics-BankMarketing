## Database Phase
import pandas as pd
import numpy as np

# Machine Learning Phase
import sklearn 
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

#Metrics Phase
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

#ignore warnings
# import warnings
# warnings.filterwarnings('ignore')

bank=pd.read_csv("./Dataset/bank-additional-full.csv",sep=';')
bank_copy=bank.copy()

# Check missing values
print("BANK DEPOSIT MARKETING ANALYTICS \n\n ")
print('\n Data columns with null values:',bank_copy.isnull().sum(), sep = '\n')

# FEATURES ENGINEERING - Handling outliers
numerical_features=['age','campaign','duration']
for cols in numerical_features:
    Q1 = bank_copy[cols].quantile(0.25)
    Q3 = bank_copy[cols].quantile(0.75)
    IQR = Q3 - Q1     

    filter = (bank_copy[cols] >= Q1 - 1.5 * IQR) & (bank_copy[cols] <= Q3 + 1.5 *IQR)
    bank_copy=bank_copy.loc[filter]

# FEATURES ENGINEERING - Education category clubbing 

bank_features=bank_copy.copy()
lst=['basic.9y','basic.6y','basic.4y']
for i in lst:
    bank_features.loc[bank_features['education'] == i, 'education'] = "middle.school"

bank_features['education'].value_counts()

# FEATURES ENGINEERING - Encoding Month and Day of week
month_dict={'may':5,'jul':7,'aug':8,'jun':6,'nov':11,'apr':4,'oct':10,'sep':9,'mar':3,'dec':12}
bank_features['month']= bank_features['month'].map(month_dict) 

day_dict={'thu':5,'mon':2,'wed':4,'tue':3,'fri':6}
bank_features['day_of_week']= bank_features['day_of_week'].map(day_dict) 

# FEATURES ENGINEERING - Encoding 999 in pdays as 0

bank_features.loc[bank_features['pdays'] == 999, 'pdays'] = 0

# FEATURES ENGINEERING - Ordinal Number Encoding

dictionary={'yes':1,'no':0,'unknown':-1}
bank_features['housing']=bank_features['housing'].map(dictionary)
bank_features['default']=bank_features['default'].map(dictionary)
bank_features['loan']=bank_features['loan'].map(dictionary)

dictionary1={'no':0,'yes':1}
bank_features['y']=bank_features['y'].map(dictionary1)

# Ordinal Encoding 


dummy_contact=pd.get_dummies(bank_features['contact'], prefix='dummy',drop_first=True)
dummy_outcome=pd.get_dummies(bank_features['poutcome'], prefix='dummy',drop_first=True)
bank_features = pd.concat([bank_features,dummy_contact,dummy_outcome],axis=1)
bank_features.drop(['contact','poutcome'],axis=1, inplace=True)

# Frequency Encoding 

bank_job=bank_features['job'].value_counts().to_dict()
bank_ed=bank_features['education'].value_counts().to_dict()

bank_features['job']=bank_features['job'].map(bank_job)
bank_features['education']=bank_features['education'].map(bank_ed)


# TARGET GUIDED ORDINAL ENCODING 
print("\n========================")
print("\nTarget Y :\n ",bank_features.groupby(['marital'])['y'].mean())

ordinal_labels=bank_features.groupby(['marital'])['y'].mean().sort_values().index
print("\nLabel Y : ",ordinal_labels)

ordinal_labels2={k:i for i,k in enumerate(ordinal_labels,0)}
print("\nOrdinal label  : ", ordinal_labels2)


bank_features['marital_ordinal']=bank_features['marital'].map(ordinal_labels2)
bank_features.drop(['marital'], axis=1,inplace=True)


#Standarisasi

print("\n ======================")
print("\n STANDARIZATION\n\n")

bank_scale=bank_features.copy()
Categorical_variables=['job', 'education', 'default', 'housing', 'loan', 'month',
       'day_of_week','y', 'dummy_telephone', 'dummy_nonexistent',
       'dummy_success', 'marital_ordinal']


feature_scale=[feature for feature in bank_scale.columns if feature not in Categorical_variables]


scaler=StandardScaler()
scaler.fit(bank_scale[feature_scale])


scaled_data = pd.concat([bank_scale[['job', 'education', 'default', 'housing', 'loan', 'month',
       'day_of_week','y', 'dummy_telephone', 'dummy_nonexistent',
       'dummy_success', 'marital_ordinal']].reset_index(drop=True),
                    pd.DataFrame(scaler.transform(bank_scale[feature_scale]), columns=feature_scale)],
                    axis=1)
print(scaled_data.head())

# FEATURE SELECTION
print("\n ======================")
print("\n FEATURE SELECTION \n\n")

X=scaled_data.drop(['y'],axis=1)
y=scaled_data.y

model = ExtraTreesClassifier()
model.fit(X,y)


# TRAIN AND TEST SPLIT (80:20)

print("\n ======================")
print("\nTRAIN AND TEST SPLIT (80:20) \n\n")

X=scaled_data.drop(['pdays','month','cons.price.idx','loan','housing','emp.var.rate','y'],axis=1)
y=scaled_data.y

X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.8,random_state=1)
print("1. Input Training:",X_train.shape)
print("2. Input Test:",X_test.shape)
print("3. Output Training:",y_train.shape)
print("4. Output Test:",y_test.shape)


# MODELLING DATA Logistic Regression with Hyper Parameter Tuning

print("\n ======================")
print("\n MODELLING DATA : Logistic Regression w/ Hyper-paramter Tuning \n\n")


param_grid = {'C': np.logspace(-4, 4, 50),
             'penalty':['l1', 'l2']}
clf = GridSearchCV(LogisticRegression(random_state=0), param_grid,cv=5, verbose=0,n_jobs=-1)
best_model = clf.fit(X_train,y_train)
print(best_model.best_estimator_)
print("The mean accuracy of the model is:",best_model.score(X_test,y_test), "\n")

logreg = LogisticRegression(C=0.5689866029018293, random_state=0)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

print("\n ======================")
print("\n EVALUATION MODEL \n\n")



confusion_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n",confusion_matrix)
print("Classification Report:\n",classification_report(y_test, y_pred))