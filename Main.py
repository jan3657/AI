import pandas as pd
import numpy as np
import graphviz
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split  # Import train_test_split function
from sklearn import metrics  # Import scikit-learn metrics module for accuracy calculation
from sklearn import preprocessing
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import pydotplus

data = pd.read_csv('./nbadata.txt')
pd.set_option('display.max_columns', 100)

awayAbbrTrans = preprocessing.LabelEncoder()
homeAbbrTrans = preprocessing.LabelEncoder()

data['gmDate'] = pd.to_numeric(data.gmDate.str.replace('-', ''))
data['awayAbbr'] = awayAbbrTrans.fit_transform(data.awayAbbr)
data['homeAbbr'] = homeAbbrTrans.fit_transform(data.homeAbbr)
data['gmSeason'] = pd.to_numeric(data.gmSeason.str.replace('-', ''))
                           
data['homeWin'] = data['homePTS'] > data['awayPTS']
data['homeWin'] = (data['homeWin'].astype(int)).replace(0, -1)
data.sort_values(by=["gmDate"])

def getHomeFilter(team, date):
    dataFilter = ((data["homeAbbr"] == team) & (data["gmDate"] <= date))
    return dataFilter
def getAwayFilter(team, date):
    dataFilter = ((data["awayAbbr"] == team) & (data["gmDate"] <= date))
    return dataFilter

array_of_arrays = []

for index, row in data.head(300).iterrows():
    if(index % 100 == 0):
        print("iter", index,)
    dataFilter = getHomeFilter(row['homeAbbr'], row['gmDate'])
    awayFilter = getAwayFilter(row['awayAbbr'], row['gmDate'])
    array_of_arrays.append([
        row['gmDate'],  # TEAM IDX
        homeAbbrTrans.inverse_transform([row['homeAbbr']])[0],
        (data.loc[dataFilter, ['homeDayOff']][-3:].mean()[0].round(2)),
        (data.loc[dataFilter, ['homePTS']][-3:].mean()[0].round(2)),  # AVG Team Home points
        (data.loc[dataFilter, ['homeAST']][-3:].mean()[0].round(2)),
        (data.loc[dataFilter, ['homeTO']][-3:].mean()[0].round(2)),
        (data.loc[dataFilter, ['homeSTL']][-3:].mean()[0].round(2)),
        (data.loc[dataFilter, ['homeBLK']][-3:].mean()[0].round(2)),
        (data.loc[dataFilter, ['homePF']][-3:].mean()[0]).round(2),
        (data.loc[dataFilter, ['homeFGA']][-3:].mean()[0].round(2)),
        (data.loc[dataFilter, ['homeFGM']][-3:].mean()[0].round(2)),
        (data.loc[dataFilter, ['home2PA']][-3:].mean()[0].round(2)),
        (data.loc[dataFilter, ['home2PM']][-3:].mean()[0].round(2)),
        (data.loc[dataFilter, ['home3PA']][-3:].mean()[0].round(2)),
        (data.loc[dataFilter, ['home3PM']][-3:].mean()[0].round(2)),
        (data.loc[dataFilter, ['homeFTA']][-3:].mean()[0].round(2)),
        (data.loc[dataFilter, ['homeFTM']][-3:].mean()[0].round(2)),
        (data.loc[dataFilter, ['homeORB']][-3:].mean()[0].round(2)),
        (data.loc[dataFilter, ['homeDRB']][-3:].mean()[0].round(2)),
        (data.loc[dataFilter, ['homeTRB']][-3:].mean()[0].round(2)),
        (data.loc[dataFilter, ['homePTS1']][-3:].mean()[0].round(2)),
        (data.loc[dataFilter, ['homePTS2']][-3:].mean()[0].round(2)),
        (data.loc[dataFilter, ['homePTS3']][-3:].mean()[0].round(2)),
        (data.loc[dataFilter, ['homePTS4']][-3:].mean()[0].round(2)),
        (data.loc[dataFilter, ['homePTSEx']][-3:].mean()[0].round(2)),

        (awayAbbrTrans.inverse_transform([row['awayAbbr']])[0]),
        (data.loc[dataFilter, ['awayDayOff']][-3:].mean()[0].round(2)),
        (data.loc[dataFilter, ['awayPTS']][-3:].mean()[0].round(2)),  # AVG Team Home points
        (data.loc[dataFilter, ['awayAST']][-3:].mean()[0].round(2)),
        (data.loc[dataFilter, ['awayTO']][-3:].mean()[0].round(2)),
        (data.loc[dataFilter, ['awaySTL']][-3:].mean()[0].round(2)),
        (data.loc[dataFilter, ['awayBLK']][-3:].mean()[0].round(2)),
        (data.loc[dataFilter, ['awayPF']][-3:].mean()[0]).round(2),
        (data.loc[dataFilter, ['awayFGA']][-3:].mean()[0].round(2)),
        (data.loc[dataFilter, ['awayFGM']][-3:].mean()[0].round(2)),
        (data.loc[dataFilter, ['away2PA']][-3:].mean()[0].round(2)),
        (data.loc[dataFilter, ['away2PM']][-3:].mean()[0].round(2)),
        (data.loc[dataFilter, ['away3PA']][-3:].mean()[0].round(2)),
        (data.loc[dataFilter, ['away3PM']][-3:].mean()[0].round(2)),
        (data.loc[dataFilter, ['awayFTA']][-3:].mean()[0].round(2)),
        (data.loc[dataFilter, ['awayFTM']][-3:].mean()[0].round(2)),
        (data.loc[dataFilter, ['awayORB']][-3:].mean()[0].round(2)),
        (data.loc[dataFilter, ['awayDRB']][-3:].mean()[0].round(2)),
        (data.loc[dataFilter, ['awayTRB']][-3:].mean()[0].round(2)),
        (data.loc[dataFilter, ['awayPTS1']][-3:].mean()[0].round(2)),
        (data.loc[dataFilter, ['awayPTS2']][-3:].mean()[0].round(2)),
        (data.loc[dataFilter, ['awayPTS3']][-3:].mean()[0].round(2)),
        (data.loc[dataFilter, ['awayPTS4']][-3:].mean()[0].round(2)),
        (data.loc[dataFilter, ['awayPTSEx']][-3:].mean()[0].round(2)),

        (data.loc[dataFilter & awayFilter, ['homeWin']].sum()[0]),
        (data.loc[dataFilter, ['homeWin']][-1:].sum()[0])



    ])

homeAvgStats = pd.DataFrame(array_of_arrays, columns=(
    ['gmDate', 'homeAbbr', 'homeDayOff', 'homePTS', 'homeAST', 'homeTO', 'homeSTL', 'homeBLK', 'homePF', 'homeFGA', 'homeFGM', 'home2PA', 'home2PM', 'home3PA', 'home3PM',
     'homeFTA','homeFTM', 'homeORB', 'homeDRB', 'homeTRB', 'homePTS1', 'homePTS2', 'homePTS3', 'homePTS4', 'homePTSEx', 'awayAbbr', 'awayDayOff', 'awayPTS', 'awayAST', 'awayTO', 'awaySTL', 'awayBLK', 'awayPF', 'awayFGA', 'awayFGM', 'away2PA', 'away2PM', 'away3PA', 'away3PM',
     'awayFTA','awayFTM', 'awayORB', 'awayDRB', 'awayTRB', 'awayPTS1', 'awayPTS2', 'awayPTS3', 'awayPTS4', 'awayPTSEx', 'homeWinDiff', 'homeWin']))

