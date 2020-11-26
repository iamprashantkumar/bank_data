import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.metrics import confusion_matrix

data=pd.read_csv('bank_data.csv')
# print(data.corr())

y=data.y
x=data[['balance','housing','loan','duration','campaign','poutfailure','poutother'
		,'poutsuccess','con_cellular','con_telephone','divorced','married','joadmin.','joblue.collar'
		,'johousemaid','jomanagement','joretired','jostudent']]
# x=data.drop('y',axis=1)
# x=x.drop('con_unknown',axis=1)
# x=x.drop('poutunknown',axis=1)
# x=x.drop('default',axis=1)
# x=x.drop('campaign',axis=1)
# x=x.drop('loan',axis=1)
# x=x.drop('married',axis=1)
# x=x.drop('joblue.collar',axis=1)
# x=x.drop('joentrepreneur',axis=1)
# x=x.drop('johousemaid',axis=1)
# x=x.drop('joservices',axis=1)
# x=x.drop('jotechnician',axis=1)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
lr=LogisticRegression(solver='lbfgs',max_iter=200)
# lr.fit(x_train,y_train)
# joblib.dump(lr,'model_joblib')
model=joblib.load('model_joblib')
predi=model.predict(x_test)
y=model.score(x_test,y_test)
print('model accuracy is = ',round(y,2))
confusion_matrix = confusion_matrix(y_test,predi)
print ('confusion_matrix\n',confusion_matrix)

