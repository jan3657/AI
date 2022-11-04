# In[0]:
import pandas as pd
import numpy as np
import graphviz
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn import preprocessing
from sklearn.tree import export_graphviz
from six import StringIO 
from IPython.display import Image  
import pydotplus

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


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test


clf_tree = DecisionTreeClassifier() # Create Decision Tree classifer object
clf_tree = clf_tree.fit(X_train,y_train) # Train Decision Tree Classifer
y_pred = clf_tree.predict(X_test) #Predict the response for test dataset


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))



dot_data = StringIO()
export_graphviz(clf_tree, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('tree.png')
Image(graph.create_png())


teams = data['homeAbbr'].unique()
home_teams = (pd.concat([
                    pd.DataFrame(
                    [
                        [
                        team, #TEAM IDX
                        homeAbbrTrans.inverse_transform([team])[0], #TEAM Name
                        (data.loc[data["homeAbbr"] == team, ['homeDayOff']].mean()[0]),
                        (data.loc[data["homeAbbr"] == team, ['homePTS']].mean()[0]), #AVG Team Home points              
                        (data.loc[data["homeAbbr"] == team, ['homeAST']].mean()[0]),    
                        (data.loc[data["homeAbbr"] == team, ['homeTO']].mean()[0]), 
                        (data.loc[data["homeAbbr"] == team, ['homeSTL']].mean()[0]),     
                        (data.loc[data["homeAbbr"] == team, ['homeBLK']].mean()[0]),     
                        (data.loc[data["homeAbbr"] == team, ['homePF']].mean()[0]),     
                        (data.loc[data["homeAbbr"] == team, ['homeFGA']].mean()[0]),
                        (data.loc[data["homeAbbr"] == team, ['homeFGM']].mean()[0]),
                        (data.loc[data["homeAbbr"] == team, ['home2PA']].mean()[0]),     
                        (data.loc[data["homeAbbr"] == team, ['home2PM']].mean()[0]),     
                        (data.loc[data["homeAbbr"] == team, ['home3PA']].mean()[0]),     
                        (data.loc[data["homeAbbr"] == team, ['home3PM']].mean()[0]),
                        (data.loc[data["homeAbbr"] == team, ['homeFTA']].mean()[0]),     
                        (data.loc[data["homeAbbr"] == team, ['homeFTM']].mean()[0]),     
                        (data.loc[data["homeAbbr"] == team, ['homeORB']].mean()[0]),     
                        (data.loc[data["homeAbbr"] == team, ['homeDRB']].mean()[0]),     
                        (data.loc[data["homeAbbr"] == team, ['homeTRB']].mean()[0]),  
                        (data.loc[data["homeAbbr"] == team, ['homePTS1']].mean()[0]), 
                        (data.loc[data["homeAbbr"] == team, ['homePTS2']].mean()[0]),     
                        (data.loc[data["homeAbbr"] == team, ['homePTS3']].mean()[0]),     
                        (data.loc[data["homeAbbr"] == team, ['homePTS4']].mean()[0]),     
                        (data.loc[data["homeAbbr"] == team, ['homePTSEx']].mean()[0])     
                        ]
                    ],
                    columns=(['AbbrIdx','Abbr','DayOff','PTS','AST','TO','STL','BLK','PF','FGA','FGM','2PA','2PM','3PA','3PM','FTA','FTM','ORB','DRB','TRB','PTS1','PTS2','PTS3','PTS4','PTSEx'])
                            
                    )
                    for team in teams
                   ],
        ignore_index=True))

away_teams = (pd.concat([
                    pd.DataFrame(
                    [
                        [
                        team, #TEAM IDX
                        awayAbbrTrans.inverse_transform([team])[0], #TEAM Name
                        (data.loc[data["awayAbbr"] == team, ['awayDayOff']].mean()[0]),
                        (data.loc[data["awayAbbr"] == team, ['awayPTS']].mean()[0]), #AVG Team away points                        
                        (data.loc[data["awayAbbr"] == team, ['awayAST']].mean()[0]),    
                        (data.loc[data["awayAbbr"] == team, ['awayTO']].mean()[0]), 
                        (data.loc[data["awayAbbr"] == team, ['awaySTL']].mean()[0]),     
                        (data.loc[data["awayAbbr"] == team, ['awayBLK']].mean()[0]),     
                        (data.loc[data["awayAbbr"] == team, ['awayPF']].mean()[0]),     
                        (data.loc[data["awayAbbr"] == team, ['awayFGA']].mean()[0]),
                        (data.loc[data["awayAbbr"] == team, ['awayFGM']].mean()[0]),
                        (data.loc[data["awayAbbr"] == team, ['away2PA']].mean()[0]),     
                        (data.loc[data["awayAbbr"] == team, ['away2PM']].mean()[0]),     
                        (data.loc[data["awayAbbr"] == team, ['away3PA']].mean()[0]),     
                        (data.loc[data["awayAbbr"] == team, ['away3PM']].mean()[0]),
                        (data.loc[data["awayAbbr"] == team, ['awayFTA']].mean()[0]),     
                        (data.loc[data["awayAbbr"] == team, ['awayFTM']].mean()[0]),     
                        (data.loc[data["awayAbbr"] == team, ['awayORB']].mean()[0]),     
                        (data.loc[data["awayAbbr"] == team, ['awayDRB']].mean()[0]),     
                        (data.loc[data["awayAbbr"] == team, ['awayTRB']].mean()[0]),  
                        (data.loc[data["awayAbbr"] == team, ['awayPTS1']].mean()[0]), 
                        (data.loc[data["awayAbbr"] == team, ['awayPTS2']].mean()[0]),     
                        (data.loc[data["awayAbbr"] == team, ['awayPTS3']].mean()[0]),     
                        (data.loc[data["awayAbbr"] == team, ['awayPTS4']].mean()[0]),     
                        (data.loc[data["awayAbbr"] == team, ['awayPTSEx']].mean()[0])     
                        ]
                    ],
                    columns=(['AbbrIdx','Abbr','DayOff','PTS','AST','TO','STL','BLK','PF','FGA','FGM','2PA','2PM','3PA','3PM','FTA','FTM','ORB','DRB','TRB','PTS1','PTS2','PTS3','PTS4','PTSEx'])
                            
                    )
                    for team in teams
                   ],
        ignore_index=True))

home_teams.set_index('Abbr', inplace=True)
away_teams.set_index('Abbr', inplace=True)



pred = clf_tree.predict(X_test) #Predict the response for test dataset


def predict(season,date,away_team,home_team):
    d = {'gmSeason' : pd.to_numeric(season.replace('-','')), 'gmDate' : pd.to_numeric(date.replace('-',''))}
    season_date = pd.Series(data=d, index=['gmSeason', 'gmDate'])
    away_team = (away_teams.loc[away_team]).add_prefix('away')
    home_team = (home_teams.loc[home_team]).add_prefix('home')
    generated_game = pd.concat([season_date,away_team,home_team])
    return(generated_game)

x = [predict(str(row['gmSeason']),str(row['gmDate']),(awayAbbrTrans.inverse_transform([row['awayAbbr']])[0]),(awayAbbrTrans.inverse_transform([row['homeAbbr']])[0])) for index, row in X_test.iterrows()]
df = (pd.concat(x,axis=1)).transpose()

y_pred = clf_tree.predict(df)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


