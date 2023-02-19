# Background
"""
Wordle is a popular puzzle currently offered daily by the New York Times. Players try to solve the puzzle by guessing a five-letter word in six tries or less, receiving feedback with every guess. For this version, each guess must be an actual word in English. Guesses that are not recognized as words by the contest are not allowed. Wordle continues to grow in popularity and versions of the game are now available in over 60 languages.
The New York Times website directions for Wordle state that the color of the tiles will change after you submit your word. A yellow tile indicates the letter in that tile is in the word, but it is in the wrong location. A green tile indicates that the letter in that tile is in the word and is in the correct location. A gray tile indicates that the letter in that tile is not included in the word at all (see Attachment 2)[2]. Figure 1 is an example solution where the correct result was found in three tries.
"""
import scipy

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


# 5. Data visualization
# 5.1. Plot the number of reported results over time
plt.figure(figsize=(12, 8))
plt.plot(df['Days'], df['res'])
plt.xlabel('Days')
plt.ylabel('Number of reported results')
plt.title('Number of reported results over time')
plt.show()

# There is a spike in the number of reported results from 10 to 70
# 5.2. Plot the number of reported results over time (zoomed in)
plt.figure(figsize=(12, 8))
plt.plot(df['Days'], df['res'])
plt.xlabel('Days')
plt.ylabel('Number of reported results')
plt.title('Number of reported results over time (zoomed in)')
plt.xlim(0, 70)
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
plt.show()

# Print the number of reported for each day after the last day in the dataset
print(df2)

