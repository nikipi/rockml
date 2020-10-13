import pandas as pd
import numpy as np
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)


url='https://raw.githubusercontent.com/datasets/investor-flow-of-funds-us/master/data/weekly.csv'
df= pd.read_csv(url)
print(df.head())


df=df.set_index('Date')
print(df.head())

# What is the type of the index?
print(df.index)

# Set the index to a DatetimeIndex type

df.index=pd.to_datetime(df.index)
print(df.index)

#  Change the frequency to monthly, sum the values and assign it to monthly.

monthly=df.resample('M').sum()
print(monthly)


# You will notice that it filled the dataFrame with months that don't have any data with NaN. Let's drop these rows


year = monthly.resample('AS-JAN').sum()
print(year)