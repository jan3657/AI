import pandas as pd
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split  # Import train_test_split function
import sklearn.feature_selection
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from itertools import combinations
from sklearn.preprocessing import PolynomialFeatures

data = pd.read_csv('./3dayAvgData.txt')
data = data.dropna()
y = data['homeWin']
data = data[[i for i in list(data.columns) if i != 'homeAbbr' and i != 'awayAbbr' and i != 'gmDate' and i != 'homeWin']]



#add interactions
combos = list(combinations(list(data.columns), 2))
colnames = list(data.columns) + ['_'.join(x) for x in combos]

poly = PolynomialFeatures(interaction_only=True, include_bias=False)
data = poly.fit_transform(data)
data = pd.DataFrame(data)
data.columns = colnames

noint_indices = [i for i, x in enumerate(list((data==0).all())) if x]
data = data.drop(data.columns[noint_indices], axis=1)


feature_cols = [i for i in list(data.columns) if i != 'homeWin']
X = data[feature_cols]
data['homeWin'] = y

#data.to_csv(r'./DataWithFeatreInteraction.txt', header=data.columns, index=None, sep=',', mode='a')

X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=70, random_state=0)
select = sklearn.feature_selection.SelectKBest( k=20)
selected_features = select.fit(X_train,y_train)
indices_selected = selected_features.get_support(indices = True)
colnames_selected = [X.columns[i] for i in indices_selected]

X_train_selected = X_train[colnames_selected]
X_test_selected = X_test[colnames_selected]

print(colnames_selected)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_hat = [x[1] for x in model.predict_proba(X_test)]
auc = roc_auc_score(y_test, y_hat)
print(auc)

model = LogisticRegression()
model.fit(X_train, y_train)
y_hat = [x[1] for x in model.predict_proba(X_test)]
auc = roc_auc_score(y_test, y_hat)
print(auc)