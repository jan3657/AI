# %%
import pandas as pd
import numpy as np
import graphviz
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn import preprocessing
from sklearn.tree import export_graphviz
from six import StringIO 
from IPython.display import Image  
import pydotplus
import sklearn_relief as sr


data = pd.read_csv('./nbadata.txt')
pd.set_option('display.max_columns',100)

awayAbbrTrans = preprocessing.LabelEncoder()
homeAbbrTrans = preprocessing.LabelEncoder()

data['gmDate'] = pd.to_numeric(data.gmDate.str.replace('-',''))
data['awayAbbr'] = awayAbbrTrans.fit_transform(data.awayAbbr)
data['homeAbbr'] = homeAbbrTrans.fit_transform(data.homeAbbr)
data['gmSeason'] = pd.to_numeric(data.gmSeason.str.replace('-',''))


data['homeWin'] = data['homePTS'] > data['awayPTS']

feature_cols = [i for i in list(data.columns) if i != 'homeWin'] #Select all colums except homeWin as features

X = data[feature_cols]

y = data['homeWin']
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

def setFilter(team,date):
    global filter 
    filter = ((data["homeAbbr"] == team) & (data["gmDate"] < date))
    return filter
    

array_of_arrays = [[
                    #row['homeAbbr'], #TEAM IDX
                    homeAbbrTrans.inverse_transform(row['homeAbbr'])[0], #TEAM Name
                    (data.loc[setFilter(row['homeAbbr'],row['gmDate']), ['homeDayOff']].mean()[0].round(2)),
                    (data.loc[filter, ['homePTS']].mean()[0].round(2)), #AVG Team Home points              
                    (data.loc[filter, ['homeAST']].mean()[0].round(2)),    
                    (data.loc[filter, ['homeTO']].mean()[0].round(2)), 
                    (data.loc[filter, ['homeSTL']].mean()[0].round(2)),     
                    (data.loc[filter, ['homeBLK']].mean()[0].round(2)),     
                    (data.loc[filter, ['homePF']].mean()[0]).round(2),     
                    (data.loc[filter, ['homeFGA']].mean()[0].round(2)),
                    (data.loc[filter, ['homeFGM']].mean()[0].round(2)),
                    (data.loc[filter, ['home2PA']].mean()[0].round(2)),     
                    (data.loc[filter, ['home2PM']].mean()[0].round(2)),     
                    (data.loc[filter, ['home3PA']].mean()[0].round(2)),     
                    (data.loc[filter, ['home3PM']].mean()[0].round(2)),
                    (data.loc[filter, ['homeFTA']].mean()[0].round(2)),     
                    (data.loc[filter, ['homeFTM']].mean()[0].round(2)),     
                    (data.loc[filter, ['homeORB']].mean()[0].round(2)),     
                    (data.loc[filter, ['homeDRB']].mean()[0].round(2)),     
                    (data.loc[filter, ['homeTRB']].mean()[0].round(2)),  
                    (data.loc[filter, ['homePTS1']].mean()[0].round(2)), 
                    (data.loc[filter, ['homePTS2']].mean()[0].round(2)),     
                    (data.loc[filter, ['homePTS3']].mean()[0].round(2)),     
                    (data.loc[filter, ['homePTS4']].mean()[0].round(2)),     
                    (data.loc[filter, ['homePTSEx']].mean()[0].round(2))  

                    ]
                   for index, row in data.iterrows()
                   #for team in teams for date in dates
                   ]
print(array_of_arrays)

#pd.DataFrame(array_of_arrays,columns=(['AbbrIdx','Abbr','DayOff','PTS','AST','TO','STL','BLK','PF','FGA','FGM','2PA','2PM','3PA','3PM','FTA','FTM','ORB','DRB','TRB','PTS1','PTS2','PTS3','PTS4','PTSEx']))




