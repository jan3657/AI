import pandas as pd
from sklearn import preprocessing

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
    dataFilter = ((data["homeAbbr"] == team) & (data["gmDate"] < date))
    return dataFilter


def getThisMatch(team, date):
    dataFilter = ((data["homeAbbr"] == team) & (data["gmDate"] == date))
    return dataFilter


def winDiffFilter(homeTeam, awayTeam, date):
    dataFilter = (data["awayAbbr"] == awayTeam) & (data["gmDate"] < date) & (data["homeAbbr"] == homeTeam)
    return dataFilter


#params
numPastMatch = 3

array_of_arrays = []

for index, row in data.iterrows():
    if index % 100 == 0:
        print("iter", index, )
    dataFilter = getHomeFilter(row['homeAbbr'], row['gmDate'])

    array_of_arrays.append([
        (data.loc[dataFilter, ['homeDayOff']][-numPastMatch:].mean()[0].round(2)),
        (data.loc[dataFilter, ['homePTS']][-numPastMatch:].mean()[0].round(2)),
        (data.loc[dataFilter, ['homeAST']][-numPastMatch:].mean()[0].round(2)),
        (data.loc[dataFilter, ['homeTO']][-numPastMatch:].mean()[0].round(2)),
        (data.loc[dataFilter, ['homeSTL']][-numPastMatch:].mean()[0].round(2)),
        (data.loc[dataFilter, ['homeBLK']][-numPastMatch:].mean()[0].round(2)),
        (data.loc[dataFilter, ['homePF']][-numPastMatch:].mean()[0]).round(2),
        (data.loc[dataFilter, ['homeFGA']][-numPastMatch:].mean()[0].round(2)),
        (data.loc[dataFilter, ['homeFGM']][-numPastMatch:].mean()[0].round(2)),
        (data.loc[dataFilter, ['home2PA']][-numPastMatch:].mean()[0].round(2)),
        (data.loc[dataFilter, ['home2PM']][-numPastMatch:].mean()[0].round(2)),
        (data.loc[dataFilter, ['home3PA']][-numPastMatch:].mean()[0].round(2)),
        (data.loc[dataFilter, ['home3PM']][-numPastMatch:].mean()[0].round(2)),
        (data.loc[dataFilter, ['homeFTA']][-numPastMatch:].mean()[0].round(2)),
        (data.loc[dataFilter, ['homeFTM']][-numPastMatch:].mean()[0].round(2)),
        (data.loc[dataFilter, ['homeORB']][-numPastMatch:].mean()[0].round(2)),
        (data.loc[dataFilter, ['homeDRB']][-numPastMatch:].mean()[0].round(2)),
        (data.loc[dataFilter, ['homeTRB']][-numPastMatch:].mean()[0].round(2)),
        (data.loc[dataFilter, ['homePTS1']][-numPastMatch:].mean()[0].round(2)),
        (data.loc[dataFilter, ['homePTS2']][-numPastMatch:].mean()[0].round(2)),
        (data.loc[dataFilter, ['homePTS3']][-numPastMatch:].mean()[0].round(2)),
        (data.loc[dataFilter, ['homePTS4']][-numPastMatch:].mean()[0].round(2)),
        (data.loc[dataFilter, ['homePTSEx']][-numPastMatch:].mean()[0].round(2)),

        (data.loc[dataFilter, ['awayDayOff']][-numPastMatch:].mean()[0].round(2)),
        (data.loc[dataFilter, ['awayPTS']][-numPastMatch:].mean()[0].round(2)),  # AVG Team Home points
        (data.loc[dataFilter, ['awayAST']][-numPastMatch:].mean()[0].round(2)),
        (data.loc[dataFilter, ['awayTO']][-numPastMatch:].mean()[0].round(2)),
        (data.loc[dataFilter, ['awaySTL']][-numPastMatch:].mean()[0].round(2)),
        (data.loc[dataFilter, ['awayBLK']][-numPastMatch:].mean()[0].round(2)),
        (data.loc[dataFilter, ['awayPF']][-numPastMatch:].mean()[0]).round(2),
        (data.loc[dataFilter, ['awayFGA']][-numPastMatch:].mean()[0].round(2)),
        (data.loc[dataFilter, ['awayFGM']][-numPastMatch:].mean()[0].round(2)),
        (data.loc[dataFilter, ['away2PA']][-numPastMatch:].mean()[0].round(2)),
        (data.loc[dataFilter, ['away2PM']][-numPastMatch:].mean()[0].round(2)),
        (data.loc[dataFilter, ['away3PA']][-numPastMatch:].mean()[0].round(2)),
        (data.loc[dataFilter, ['away3PM']][-numPastMatch:].mean()[0].round(2)),
        (data.loc[dataFilter, ['awayFTA']][-numPastMatch:].mean()[0].round(2)),
        (data.loc[dataFilter, ['awayFTM']][-numPastMatch:].mean()[0].round(2)),
        (data.loc[dataFilter, ['awayORB']][-numPastMatch:].mean()[0].round(2)),
        (data.loc[dataFilter, ['awayDRB']][-numPastMatch:].mean()[0].round(2)),
        (data.loc[dataFilter, ['awayTRB']][-numPastMatch:].mean()[0].round(2)),
        (data.loc[dataFilter, ['awayPTS1']][-numPastMatch:].mean()[0].round(2)),
        (data.loc[dataFilter, ['awayPTS2']][-numPastMatch:].mean()[0].round(2)),
        (data.loc[dataFilter, ['awayPTS3']][-numPastMatch:].mean()[0].round(2)),
        (data.loc[dataFilter, ['awayPTS4']][-numPastMatch:].mean()[0].round(2)),
        (data.loc[dataFilter, ['awayPTSEx']][-numPastMatch:].mean()[0].round(2)),
        row['gmDate'] % 10000 > 1000 or row['gmDate'] % 10000 < 220 ,
        int((data.loc[winDiffFilter(row['homeAbbr'], row['awayAbbr'], row['gmDate']), ['homeWin']].sum()[0])),
        (data.loc[getThisMatch(row['homeAbbr'], row['gmDate']), ['homeWin']][-1:].sum()[0]) == 1

    ])

homeAvgStats = pd.DataFrame(array_of_arrays, columns=(
    ['homeDayOff', 'homePTS', 'homeAST', 'homeTO', 'homeSTL', 'homeBLK', 'homePF', 'homeFGA',
     'homeFGM', 'home2PA', 'home2PM', 'home3PA', 'home3PM',
     'homeFTA', 'homeFTM', 'homeORB', 'homeDRB', 'homeTRB', 'homePTS1', 'homePTS2', 'homePTS3', 'homePTS4', 'homePTSEx',
     'awayDayOff', 'awayPTS', 'awayAST', 'awayTO', 'awaySTL', 'awayBLK', 'awayPF', 'awayFGA', 'awayFGM',
     'away2PA', 'away2PM', 'away3PA', 'away3PM',
     'awayFTA', 'awayFTM', 'awayORB', 'awayDRB', 'awayTRB', 'awayPTS1', 'awayPTS2', 'awayPTS3', 'awayPTS4', 'awayPTSEx',
     'preASB', 'homeWinDiff', 'homeWin']))

homeAvgStats.to_csv(r'./3dayAvgData.txt', header=homeAvgStats.columns, index=None, sep=',', mode='a')

