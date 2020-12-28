#import codecademylib3_seaborn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# load and investigate the data here:

tennis_df = pd.read_csv('tennis_stats.csv')


# perform exploratory analysis here:

print(tennis_df.columns)

correlation = tennis_df.corr()
correlation.to_csv('correlation.csv')

# Result from Correlation : 
    #BreakPointsOpportunities : 0.923453317366972 - number of times where the player could have won the service game of the opponent
    #ReturnGamesPlayed : 0.928407622580664 - total number of games where the player’s opponent served
    #ServiceGamesPlayed : 0.929152535824128 - total number of games where the player served
    #DoubleFaults	0.847459335 - number of times player missed both first and second serve attempts
    #Aces	0.825301192 - number of serves by the player where the receiver does not touch the ball

plt.scatter(tennis_df.BreakPointsOpportunities, tennis_df.Wins)
plt.show()
plt.scatter(tennis_df.ReturnGamesPlayed, tennis_df.Wins)
plt.show()
plt.scatter(tennis_df.ServiceGamesPlayed, tennis_df.Wins)
plt.show()
plt.scatter(tennis_df.DoubleFaults, tennis_df.Wins)
plt.show()
plt.scatter(tennis_df.Aces, tennis_df.Wins)
plt.show()


#perform single feature linear regressions here - with BreakPointsOpportunities :
x_break = tennis_df[['BreakPointsOpportunities']]
y = tennis_df[['Wins']]
x_break_train, x_break_test, y_train, y_test = train_test_split(x_break, y, train_size = 0.8)

model = LinearRegression()
model.fit(x_break_train,y_train)

scoring = model.score(x_break_train, y_train)
prediction = model.predict(x_break_test)
plt.scatter(x_break_test,prediction, alpha=0.4)

plt.show()


## perform single feature linear regressions here - with Aces :
x_aces = tennis_df[['Aces']]
y = tennis_df[['Wins']]
x_aces_train, x_aces_test, y_train, y_test = train_test_split(x_aces, y, train_size = 0.8)

model = LinearRegression()
model.fit(x_aces_train,y_train)

scoring = model.score(x_aces_train, y_train)
prediction = model.predict(x_aces_test)
plt.scatter(x_aces_test,prediction, alpha=0.4)

plt.show()


## perform two feature linear regressions here:

x_two_features = tennis_df[['ServiceGamesPlayed', 'ReturnGamesPlayed']]
y = tennis_df[['Wins']]

x_two_features_train, x_two_features_test, y_train, y_test = train_test_split(x_two_features, y, train_size = 0.8)

model = LinearRegression()
model.fit(x_two_features_train,y_train)

scoring = model.score(x_two_features_train, y_train)
print(scoring)
prediction = model.predict(x_two_features_test)
plt.scatter(x_two_features_test,prediction, alpha=0.4)

plt.show()

## perform multiple feature linear regressions here:

features = tennis_df[['ServiceGamesPlayed', 'ReturnGamesPlayed', 'BreakPointsOpportunities', 'DoubleFaults', 'Aces']]
outcome = tennis_df[['Wins']]

features_train, features_test, outcome_train, outcome_test = train_test_split(features, outcome, train_size = 0.8)
model = LinearRegression()
model.fit(features_train,outcome_train)

scoring = model.score(features_train, outcome_train)
print(scoring)

prediction = model.predict(features_test)

plt.scatter(features_test,prediction)

plt.show()