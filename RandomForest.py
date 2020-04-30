from sklearn import datasets
iris=datasets.load_iris()

print(iris.target_names)
print(iris.feature_names)
print(iris.data[0:5])
print(iris.target)

import pandas as pd
data=pd.DataFrame({
    'sepal length':iris.data[:,0],
    'sepal width':iris.data[:,1],
    'petal length':iris.data[:,2],
    'petal width':iris.data[:,3],
    'species':iris.target
})
data.head()
print(data.head())

from sklearn.model_selection import train_test_split
X=data[['sepal length', 'sepal width', 'petal length', 'petal width']]
y=data['species']
X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.25, random_state=101)

from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=100, random_state=101)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)

from sklearn import metrics
print("accurcy", metrics.accuracy_score(y_test, y_pred))
print(clf.fit(X_train,y_train))

feature_imp=pd.Series(clf.feature_importances_,index=iris.feature_names).sort_values(ascending=False)
print(feature_imp)

import matplotlib.pyplot as plt
import seaborn as sns
sns.barplot(x=feature_imp,y=feature_imp.index)
plt.xlabel('feature important score')
plt.ylabel('features')
plt.title('visualizing important features')
plt.show()