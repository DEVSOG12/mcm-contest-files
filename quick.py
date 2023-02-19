# Background
"""
Wordle is a popular puzzle currently offered daily by the New York Times. Players try to solve the puzzle by guessing a five-letter word in six tries or less, receiving feedback with every guess. For this version, each guess must be an actual word in English. Guesses that are not recognized as words by the contest are not allowed. Wordle continues to grow in popularity and versions of the game are now available in over 60 languages.
The New York Times website directions for Wordle state that the color of the tiles will change after you submit your word. A yellow tile indicates the letter in that tile is in the word, but it is in the wrong location. A green tile indicates that the letter in that tile is in the word and is in the correct location. A gray tile indicates that the letter in that tile is not included in the word at all (see Attachment 2)[2]. Figure 1 is an example solution where the correct result was found in three tries.
"""
import datetime

import scipy
from sklearn.linear_model import LinearRegression
from textstat import textstat
from textblob import TextBlob
from wordfreq import word_frequency

"""
Players can play in regular mode or “Hard Mode.” Wordle’s Hard Mode makes the game more difficult by requiring that once a player has found a correct letter in a word (the tile is yellow or green), those letters must be used in subsequent guesses. The example in Figure 1 was played in Hard Mode.
Many (but not all) users report their scores on Twitter. For this problem, MCM has generated a file of daily results for January 7, 2022 through December 31, 2022 (see Attachment 1). This file includes the date, contest number, word of the day, the number of people reporting scores that day, the number of players on hard mode, and the percentage that guessed the word in one try, two tries, three tries, four tries, five tries, six tries, or could not solve the puzzle (indicated by X). For example, in Figure 2 the word on July 20, 2022 was “TRITE” and the results were obtained by mining Twitter. Although the percentages in Figure 2 sum to 100%, in some cases this may not be true due to rounding.
"""

# Notes
# Attachment 1 is data.csv

# Requirements
"""
The number of reported results vary daily. Develop a model to explain this variation and use your model to create a prediction interval for the number of reported results on March 1, 2023
"""

# Solution
# 1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
import scipy.stats as stats

# 2. Import data
df = pd.read_csv('data.csv')

# 3. Explore data
df.head()
df.info()
df.describe()

"""
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 359 entries, 0 to 358
Data columns (total 12 columns):
 #   Column               Non-Null Count  Dtype 
---  ------               --------------  ----- 
 0   Date                 359 non-null    object
 1   Contest Number       359 non-null    int64 
 2   word                 359 non-null    object
 3   res                  359 non-null    int64 
 4   Number in Hard Mode  359 non-null    int64 
 5   1 try %              359 non-null    int64 
 6   2 tries %            359 non-null    int64 
 7   3 tries %            359 non-null    int64 
 8   4 tries %            359 non-null    int64 
 9   5 tries %            359 non-null    int64 
 10  6 tries %            359 non-null    int64 
 11  7 or more tries %    359 non-null    int64 
dtypes: int64(10), object(2)
"""

# 4. Data cleaning
# 4.1. Convert Date to datetime
df['Date'] = pd.to_datetime(df['Date'])

# 4.2. Create a new column with the number of days since the first date
df['Days'] = (df['Date'] - df['Date'].min()).dt.days

# Cut off the first 5 months of data
df = df[df['Days'] > 150]


# 5. Data visualization
# 5.1. Plot the number of reported results over time
plt.figure(figsize=(12, 8))
plt.plot(df['Days'], df['res'])
plt.xlabel('Days')
plt.ylabel('Number of reported results')
plt.title('Number of reported results over time')
plt.savefig('/Users/oreofe/PycharmProjects/DataCScub/Images/number_of_reported_results_over_time.png')
plt.show()

# There is a spike in the number of reported results from 10 to 70
# 5.2. Plot the number of reported results over time (zoomed in)
plt.figure(figsize=(12, 8))
plt.plot(df['Days'], df['res'])
plt.xlabel('Days')
plt.ylabel('Number of reported results')
plt.title('Number of reported results over time (zoomed in)')
plt.xlim(0, 70)
plt.savefig('/Users/oreofe/PycharmProjects/DataCScub/Images/number_of_reported_results_over_time_zoomed_in.png')
plt.show()



# 6. Model
# 6.1. Create a model
model = smf.ols('res ~ Days', data=df).fit()

# 6.2. Model summary
print(model.summary())


"""
 OLS Regression Results                            
==============================================================================
Dep. Variable:                    res   R-squared:                       0.675
Model:                            OLS   Adj. R-squared:                  0.674
Method:                 Least Squares   F-statistic:                     742.6
Date:                Sun, 19 Feb 2023   Prob (F-statistic):           3.17e-89
Time:                        15:26:08   Log-Likelihood:                -4399.4
No. Observations:                 359   AIC:                             8803.
Df Residuals:                     357   BIC:                             8811.
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept   2.175e+05   5365.731     40.528      0.000    2.07e+05    2.28e+05
Days        -706.9329     25.942    -27.251      0.000    -757.951    -655.915
==============================================================================
Omnibus:                       31.648   Durbin-Watson:                   0.064
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               37.479
Skew:                           0.745   Prob(JB):                     7.27e-09
Kurtosis:                       3.536   Cond. No.                         413.
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
"""

# Optimized model
model = smf.ols('res ~ Days + np.power(Days, 2)', data=df).fit()

# Model summary
print(model.summary())



# Plot
plt.figure(figsize=(12, 8))
plt.plot(df['Days'], df['res'], 'o')
plt.plot(df['Days'], model.predict(df['Days']), 'r')
plt.xlabel('Days')
plt.ylabel('Number of reported results')
plt.title('Number of reported results over time')
plt.savefig('/Users/oreofe/PycharmProjects/DataCScub/Images/number_of_reported_results_over_time_with_model.png')
plt.show()




# 6.3. Model diagnostics
# 6.3.1. Normality
#
# The residuals are not normally distributed
#
# Shapiro-Wilk test
print(stats.shapiro(model.resid))

"""
ShapiroResult(statistic=0.7801181674003601, pvalue=1.207792847433154e-21)
"""

# QQ plot
fig = sm.qqplot(model.resid, line='s')
plt.savefig('/Users/oreofe/PycharmProjects/DataCScub/Images/qq_plot.png')
plt.show()

# 6.3.2. Homoscedasticity
#
# The residuals are not homoscedastic

# Plot
plt.figure(figsize=(12, 8))
plt.plot(model.predict(df['Days']), model.resid, 'o')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.title('Residuals vs. predicted values')
plt.savefig('/Users/oreofe/PycharmProjects/DataCScub/Images/residuals_vs_predicted_values.png')
plt.show()

# Optimize model to fix homoscedasticity and normality
model = smf.ols('res ~ Days + np.power(Days, 2) + np.power(Days, 3)', data=df).fit()

# Model summary
print(model.summary())

"""
                          OLS Regression Results                            
==============================================================================
Dep. Variable:                    res   R-squared:                       0.823
Model:                            OLS   Adj. R-squared:                  0.822
Method:                 Least Squares   F-statistic:                     551.8
Date:                Sun, 19 Feb 2023   Prob (F-statistic):          2.91e-133
Time:                        15:31:22   Log-Likelihood:                -4290.0
No. Observations:                 359   AIC:                             8588.
Df Residuals:                     355   BIC:                             8604.
Df Model:                           3                                         
Covariance Type:            nonrobust                                         
=====================================================================================
                        coef    std err          t      P>|t|      [0.025      0.975]
-------------------------------------------------------------------------------------
Intercept          2.938e+05   7870.555     37.328      0.000    2.78e+05    3.09e+05
Days              -1994.1226    190.660    -10.459      0.000   -2369.087   -1619.158
np.power(Days, 2)     3.6316      1.239      2.932      0.004       1.196       6.067
np.power(Days, 3)    -0.0001      0.002     -0.049      0.961      -0.005       0.004
==============================================================================
Omnibus:                      142.266   Durbin-Watson:                   0.119
Prob(Omnibus):                  0.000   Jarque-Bera (JB):             1561.468
Skew:                          -1.332   Prob(JB):                         0.00
Kurtosis:                      12.863   Cond. No.                     6.89e+07
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
[2] The condition number is large, 6.89e+07. This might indicate that there are
strong multicollinearity or other numerical problems.
"""

# Plot
plt.figure(figsize=(12, 8))
plt.plot(df['Days'], df['res'], 'o')
plt.plot(df['Days'], model.predict(df['Days']), 'r')
plt.xlabel('Days')
plt.ylabel('Number of reported results')
plt.title('Number of reported results over time')
plt.savefig('/Users/oreofe/PycharmProjects/DataCScub/Images/number_of_reported_results_over_time_with_model_homosdk.png')
plt.show()


dfk = pd.DataFrame({'Days': range(df['Days'].min() + 1, df['Days'].max() + 1 + 30)})
dfk['res'] = model.predict(dfk['Days'])

# Plot
plt.figure(figsize=(12, 8))
plt.plot(df['Days'], df['res'], 'o')
plt.plot(dfk['Days'], dfk['res'], 'o')
plt.xlabel('Days')
plt.ylabel('Number of reported results')
plt.title('Number of reported results over time')
plt.savefig('/Users/oreofe/PycharmProjects/DataCScub/Images/number_of_reported_results_over_time_with_predictions_n.png')
plt.show()

# Let's run predictions for the next x days till March 1st after the last day in the dataset
# We will use the optimized model

# Create a dataframe with the next x days
df2 = pd.DataFrame({'Days': range(df['Days'].max() + 1, df['Days'].max() + 1 + 30)})
df2['res'] = model.predict(df2['Days'])

# Plot
plt.figure(figsize=(12, 8))
plt.plot(df['Days'], df['res'], 'o')
plt.plot(df2['Days'], df2['res'], 'o')
plt.xlabel('Days')
plt.ylabel('Number of reported results')
plt.title('Number of reported results over time')
plt.savefig('/Users/oreofe/PycharmProjects/DataCScub/Images/number_of_reported_results_over_time_with_predictions.png')
plt.show()


# Print the number of reported for each day after the last day in the dataset
# Adjust the number of days to print to MM/DD/YYYY format. Note that the first day is 01/31/2022
for i in range(0, 30):
    print('Day', df['Days'].max() + 1 + i, '(', datetime.date(2022, 1, 31) + datetime.timedelta(days=i), '):', int(df2.iloc[i]['res']))



# 7. Predictions Interval
# 7.1. Confidence Interval
# 7.1.1. Confidence Interval for the mean

# Calculate the confidence interval for the mean
print(model.conf_int(alpha=0.05, cols=None))

"""
                               0              1
Intercept          278311.789741  309269.339471
Days                -2369.086860   -1619.158259
np.power(Days, 2)       1.195843       6.067441
np.power(Days, 3)      -0.004584       0.004360
"""

# 7.1.2. Confidence Interval for the prediction

# Calculate the confidence interval for the prediction
print(model.get_prediction(df2['Days']).summary_frame(alpha=0.05))

"""
            mean       mean_se  ...  obs_ci_lower   obs_ci_upper
0   40765.277843   8036.712414  ... -34989.350419  116519.906106
1   41338.856939   8205.875986  ... -34485.882054  117163.595932
2   41919.457265   8378.040401  ... -33978.067028  117816.981558
3   42507.078147   8553.201337  ... -33465.975351  118480.131646
4   43101.718915   8731.355043  ... -32949.677937  119153.115767
5   43703.378895   8912.498340  ... -32429.246590  119836.004380
6   44312.057414   9096.628606  ... -31904.753995  120528.868824
7   44927.753802   9283.743774  ... -31376.273697  121231.781301
8   45550.467385   9473.842314  ... -30843.880090  121944.814859
9   46180.197491   9666.923229  ... -30307.648397  122668.043379
10  46816.943447   9862.986035  ... -29767.654657  123401.541551
11  47460.704581  10062.030753  ... -29223.975698  124145.384861
12  48111.480222  10264.057894  ... -28676.689125  124899.649569
13  48769.269695  10469.068444  ... -28125.873295  125664.412686
14  49434.072330  10677.063849  ... -27571.607296  126439.751956
15  50105.887454  10888.046002  ... -27013.970923  127225.745830
16  50784.714394  11102.017229  ... -26453.044657  128022.473444
17  51470.552477  11318.980274  ... -25888.909636  128830.014591
18  52163.401033  11538.938282  ... -25321.647634  129648.449700
19  52863.259387  11761.894789  ... -24751.341029  130477.859804
20  53570.126869  11987.853708  ... -24178.072779  131318.326516
21  54284.002804  12216.819309  ... -23601.926391  132169.931999
22  55004.886522  12448.796216  ... -23022.985891  133032.758935
23  55732.777350  12683.789385  ... -22441.335796  133906.890495
24  56467.674615  12921.804095  ... -21857.061080  134792.410310
25  57209.577645  13162.845937  ... -21270.247144  135689.402434
26  57958.485767  13406.920800  ... -20680.979781  136597.951315
27  58714.398310  13654.034859  ... -20089.345140  137518.141760
28  59477.314601  13904.194567  ... -19495.429698  138450.058899
29  60247.233967  14157.406640  ... -18899.320218  139393.788152

[30 rows x 6 columns]
"""

# 7.2. Prediction Interval

# Calculate the prediction interval
print(model.get_prediction(df2['Days']).summary_frame(alpha=0.05))

"""
[30 rows x 6 columns]
            mean       mean_se  ...  obs_ci_lower   obs_ci_upper
0   40765.277843   8036.712414  ... -34989.350419  116519.906106
1   41338.856939   8205.875986  ... -34485.882054  117163.595932
2   41919.457265   8378.040401  ... -33978.067028  117816.981558
3   42507.078147   8553.201337  ... -33465.975351  118480.131646
4   43101.718915   8731.355043  ... -32949.677937  119153.115767
5   43703.378895   8912.498340  ... -32429.246590  119836.004380
6   44312.057414   9096.628606  ... -31904.753995  120528.868824
7   44927.753802   9283.743774  ... -31376.273697  121231.781301
8   45550.467385   9473.842314  ... -30843.880090  121944.814859
9   46180.197491   9666.923229  ... -30307.648397  122668.043379
10  46816.943447   9862.986035  ... -29767.654657  123401.541551
11  47460.704581  10062.030753  ... -29223.975698  124145.384861
12  48111.480222  10264.057894  ... -28676.689125  124899.649569
13  48769.269695  10469.068444  ... -28125.873295  125664.412686
14  49434.072330  10677.063849  ... -27571.607296  126439.751956
15  50105.887454  10888.046002  ... -27013.970923  127225.745830
16  50784.714394  11102.017229  ... -26453.044657  128022.473444
17  51470.552477  11318.980274  ... -25888.909636  128830.014591
18  52163.401033  11538.938282  ... -25321.647634  129648.449700
19  52863.259387  11761.894789  ... -24751.341029  130477.859804
20  53570.126869  11987.853708  ... -24178.072779  131318.326516
21  54284.002804  12216.819309  ... -23601.926391  132169.931999
22  55004.886522  12448.796216  ... -23022.985891  133032.758935
23  55732.777350  12683.789385  ... -22441.335796  133906.890495
24  56467.674615  12921.804095  ... -21857.061080  134792.410310
25  57209.577645  13162.845937  ... -21270.247144  135689.402434
26  57958.485767  13406.920800  ... -20680.979781  136597.951315
27  58714.398310  13654.034859  ... -20089.345140  137518.141760
28  59477.314601  13904.194567  ... -19495.429698  138450.058899
29  60247.233967  14157.406640  ... -18899.320218  139393.788152

[30 rows x 6 columns]
"""

# Explain the results
print(model.get_prediction(df2['Days']).summary_frame(alpha=0.05).describe())

"""
               mean       mean_se  ...  obs_ci_lower   obs_ci_upper
count     30.000000     30.000000  ...     30.000000      30.000000
mean   50031.766919  10894.279017  ... -27168.739114  127232.272952
std     5918.939432   1860.589827  ...   4894.576967    6944.549852
min    40765.277843   8036.712414  ... -34989.350419  116519.906106
25%    45083.432198   9331.268409  ... -31243.175295  121410.039690
50%    49769.979892  10782.554925  ... -27292.789109  126832.748893
75%    54824.665593  12390.801989  ... -23167.721016  132817.052201
max    60247.233967  14157.406640  ... -18899.320218  139393.788152

[8 rows x 6 columns]
"""

print("PREDICTIONS ON MARCH 1")
# Give be exact prediction interval at the last day of the data
print(model.get_prediction(df2['Days']).summary_frame(alpha=0.05).tail(1))
# And Explain the results
print(model.get_prediction(df2['Days']).summary_frame(alpha=0.05).tail(1).describe())

# Plot the predictions interval on the data
# Plot dfk
dfk = pd.DataFrame({'Days': range(df['Days'].min() + 1, df['Days'].max() + 1 + 30)})
dfk['res'] = model.predict(dfk['Days'])

# Make sure lower and upper bounds are not negative
# dfk['res'] = dfk['res'].apply(lambda x: 0 if x < 0 else x)
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(dfk['Days'], dfk['res'], 'o', label='data')
# ax.plot(dfk['Days'], model., 'r--.', label='OLS')
ax.plot(dfk['Days'], model.predict(dfk['Days']), 'g', label='Predictions')
# ax.plot(dfk['Days'], model.get_prediction(dfk['Days']).summary_frame(alpha=0.05)['obs_ci_lower'], 'r--', label='Predictions') # Remove the negative values from the lower bound

ax.plot(dfk['Days'], model.get_prediction(dfk['Days']).summary_frame(alpha=0.05)['obs_ci_upper'], 'r--', label='Prediction Interval')
ax.set_xlabel('Days')
ax.set_ylabel('Number of Cases')
ax.legend(loc='best')
plt.savefig('prediction.png')
plt.show()






# 8 Do any attributes of the word affect the percentage of scores reported that were played in Hard Mode?

# 8.1. Data Preparation

# Create a new dataframe
df3 = df[['word', 'Number in Hard Mode', 'res'] ]
print(df3)

"""
      word  Number in Hard Mode
0    manly                 1899
1    molar                 1973
2    havoc                 1919
3    impel                 1937
4    condo                 2012
..     ...                  ...
354  drink                 3017
355  query                 2242
356  gorge                 1913
357  crank                 1763
358  slump                 1362

[359 rows x 2 columns]
"""

# 8.2. Data Analysis

# Create a new column for the percentage of scores reported that were played in Hard Mode
df3.loc[:, 'Percentage in Hard Mode'] = df3['Number in Hard Mode'] / df3['res'] * 100
# df3.loc[:, 'column_name'] = df3['Percentage in Hard Mode'].apply(lambda x: round(x, 1))


# Create a new dataframe for the top 10 words
df3_10 = df3.sort_values(by='Percentage in Hard Mode', ascending=False).head(10)

# Create a new dataframe for the bottom 10 words
df3_10_bottom = df3.sort_values(by='Percentage in Hard Mode', ascending=True).head(10)


# 8.3. Data Visualization

# Create a bar chart for the top 10 words
plt.figure(figsize=(10, 5))
plt.bar(df3_10['word'], df3_10['Percentage in Hard Mode'])
plt.title('Top 10 Words with the Highest Percentage of Scores Reported that Were Played in Hard Mode')
plt.xlabel('Word')
plt.ylabel('Percentage in Hard Mode')
plt.savefig('/Users/oreofe/PycharmProjects/DataCScub/Images/Top_10_Words_with_the_Highest_Percentage_of_Scores_Reported_that_Were_Played_in_Hard_Mode.png')
plt.show()


# Top 10 Words with the Highest Percentage of Scores Reported that Were Played in Hard Mode
# Study - 90.0%
# Piney 18.0%
# Parer 18.0%
# Mummy 18.0%
# Catch 17.0%
# Judge 17.0%
# Waltz 17.0%
# Ionic 17.0%
# Libel 17.0%
# Extra 17.0%


# Create a bar chart for the bottom 10 words
plt.figure(figsize=(10, 5))
plt.bar(df3_10_bottom['word'], df3_10_bottom['Percentage in Hard Mode'])
plt.title('Top 10 Words with the Lowest Percentage of Scores Reported that Were Played in Hard Mode')
plt.xlabel('Word')
plt.ylabel('Percentage in Hard Mode')
plt.savefig('/Users/oreofe/PycharmProjects/DataCScub/Images/Top_10_Words_with_the_Lowest_Percentage_of_Scores_Reported_that_Were_Played_in_Hard_Mode.png')
plt.show()


# Top 10 Words with the Lowest Percentage of Scores Reported that Were Played in Hard Mode
# Robin 1.2%
# Slump 1.6%
# Crank 1.7%
# Drink 1.8%
# Gorge 2.0%
# Query 2.1%
# Favor 2.3%
# Panic 2.3%
# Tangy 2.4%
# Solar 2.4%


# Create a new column for the number of vowels in the word
df3['vowels'] = df3['word'].str.count('[aeiou]')

# Create a new column for the frequency of the word in the English language
df3['freq'] = df3['word'].apply(lambda x: word_frequency(x, 'en', wordlist='large'))

# Create a new column for the number of syllables in the word
df3['syllables'] = df3['word'].apply(textstat.syllable_count)

# Create a new column for the part of speech of the word using from textblob import TextBlob
df3['pos'] = df3['word'].apply(lambda x: TextBlob(x).tags[0][1] )
# Convert the part of speech to a number
df3['pos'] = df3['pos'].apply(lambda x: 1 if x == 'NN' else 2 if x == 'VB' else 3 if x == 'JJ' else 4 if x == 'RB' else 5 if x == 'PRP' else 6 if x == 'DT' else 7 if x == 'IN' else 8 if x == 'CC' else 9 if x == 'CD' else 10 if x == 'NNS' else 11 if x == 'VBD' else 12 if x == 'VBG' else 13 if x == 'VBN' else 14 if x == 'VBP' else 15 if x == 'VBZ' else 16 if x == 'JJR' else 17 if x == 'JJS' else 18 if x == 'RBR' else 19 if x == 'RBS' else 20 if x == 'PRP$' else 21 if x == 'WP' else 22 if x == 'WP$' else 23 if x == 'MD' else 24 if x == 'EX' else 25 if x == 'WDT' else 26 if x == 'PDT' else 27 if x == 'RP' else 28 if x == 'FW' else 29 if x == 'UH' else 30 if x == 'SYM' else 31 if x == 'TO' else 32 if x == 'LS' else 33 if x == 'POS' else 34 if x == 'NNP' else 35 if x == 'NNPS' else 36 if x == 'WRB' else 37 if x == 'NNPS' else 0)

df3 = df3[['word', 'Number in Hard Mode', 'res', 'pos', 'vowels', 'freq', 'syllables'] ]

# 9.2. Data Analysis

# Create a new column for the percentage of scores reported that were played in Hard Mode
df3['Percentage in Hard Mode'] = df3['Number in Hard Mode'] / df3['res'] * 100

# We want to see if the percentage of scores reported that were played in Hard Mode is correlated with the number of vowels in the word, the frequency of the word in the English language, the number of syllables in the word, and the part of speech of the word.

# Let's create a correlation matrix
print(df3.corr())

"""
                        Number in Hard Mode  ...  Percentage in Hard Mode
Number in Hard Mode                 1.000000  ...                -0.370842
res                                 0.922252  ...                -0.435233
pos                                 0.116729  ...                -0.043300
vowels                             -0.029404  ...                -0.027399
freq                                0.137446  ...                -0.016299
syllables                          -0.100939  ...                 0.013570
Percentage in Hard Mode            -0.370842  ...                 1.000000
"""

# We can see that the percentage of scores reported that were played in Hard Mode is not correlated with the number of vowels in the word, the frequency of the word in the English language, the number of syllables in the word, and the part of speech of the word.

# 9.3. Data Visualization

# Create a scatter plot for the number of vowels in the word and the percentage of scores reported that were played in Hard Mode
plt.figure(figsize=(10, 5))
plt.scatter(df3['vowels'], df3['Percentage in Hard Mode'])
plt.title('Number of Vowels in the Word vs. Percentage of Scores Reported that Were Played in Hard Mode')
plt.xlabel('Number of Vowels in the Word')
plt.ylabel('Percentage in Hard Mode')
plt.savefig('/Users/oreofe/PycharmProjects/DataCScub/Images/Number_of_Vowels_in_the_Word_vs_Percentage_of_Scores_Reported_that_Were_Played_in_Hard_Mode.png')
plt.show()


# Number of Vowels in the Word vs. Percentage of Scores Reported that Were Played in Hard Mode
# There is no correlation between the number of vowels in the word and the percentage of scores reported that were played in Hard Mode.


# Create a scatter plot for the frequency of the word in the English language and the percentage of scores reported that were played in Hard Mode
plt.figure(figsize=(10, 5))
plt.scatter(df3['freq'], df3['Percentage in Hard Mode'])
plt.title('Frequency of the Word in the English Language vs. Percentage of Scores Reported that Were Played in Hard Mode')
plt.xlabel('Frequency of the Word in the English Language')
plt.ylabel('Percentage in Hard Mode')
plt.savefig('/Users/oreofe/PycharmProjects/DataCScub/Images/Frequency_of_the_Word_in_the_English_Language_vs_Percentage_of_Scores_Reported_that_Were_Played_in_Hard_Mode.png')
plt.show()


# Frequency of the Word in the English Language vs. Percentage of Scores Reported that Were Played in Hard Mode
# There is no correlation between the frequency of the word in the English language and the percentage of scores reported that were played in Hard Mode.


# Create a scatter plot for the number of syllables in the word and the percentage of scores reported that were played in Hard Mode
plt.figure(figsize=(10, 5))
plt.scatter(df3['syllables'], df3['Percentage in Hard Mode'])
plt.title('Number of Syllables in the Word vs. Percentage of Scores Reported that Were Played in Hard Mode')
plt.xlabel('Number of Syllables in the Word')
plt.ylabel('Percentage in Hard Mode')
plt.savefig('/Users/oreofe/PycharmProjects/DataCScub/Images/Number_of_Syllables_in_the_Word_vs_Percentage_of_Scores_Reported_that_Were_Played_in_Hard_Mode.png')
plt.show()


# Number of Syllables in the Word vs. Percentage of Scores Reported that Were Played in Hard Mode
# There is no correlation between the number of syllables in the word and the percentage of scores reported that were played in Hard Mode.


# Create a scatter plot for the part of speech of the word and the percentage of scores reported that were played in Hard Mode
plt.figure(figsize=(10, 5))
plt.scatter(df3['pos'], df3['Percentage in Hard Mode'])
plt.title('Part of Speech of the Word vs. Percentage of Scores Reported that Were Played in Hard Mode')
plt.xlabel('Part of Speech of the Word')
plt.ylabel('Percentage in Hard Mode')
plt.savefig('/Users/oreofe/PycharmProjects/DataCScub/Images/Part_of_Speech_of_the_Word_vs_Percentage_of_Scores_Reported_that_Were_Played_in_Hard_Mode.png')
plt.show()


# Do any attributes of the word affect the percentage of scores reported that were played in Hard Mode? If so, how? If not, why not?

"""
From the scatter plots, we can see that the number of vowels in the word, the frequency of the word in the English language, the number of syllables in the word, and the part of speech of the word do not affect the percentage of scores reported that were played in Hard Mode.

This is because the percentage of scores reported that were played in Hard Mode is not correlated with the number of vowels in the word, the frequency of the word in the English language, the number of syllables in the word, and the part of speech of the word.

The percentage of scores reported that were played in Hard Mode is negatively correlated with the number of scores reported for the word, which means that the more scores that were reported for the word, the less likely it is that the percentage of scores reported that were played in Hard Mode is high.

This is because the more scores that were reported for the word, the more likely it is that the percentage of scores reported that were played in Hard Mode is low.
"""


# For a given future solution word on a future date, develop a model that allows you to predict the distribution of the reported results. In other words, to predict the associated percentages of (1, 2, 3, 4, 5, 6, X) for a future date.

# 10.1. Data Preprocessing

# Create a new column for the number of scores reported for the word
df3['Number of Scores Reported'] = df3['res']

# Only two modes (Regular and Hard) are available in the game, so the percentage of scores reported that were played in Regular Mode is 100% - the percentage of scores reported that were played in Hard Mode
df3['Percentage in Regular Mode'] = 100 - df3['Percentage in Hard Mode']


# 10.2. Data Analysis

# Create a new column for the percentage of scores reported that were played in Regular Mode
