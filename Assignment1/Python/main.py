import pandas as pd
import numpy as np

df = pd.read_csv("../Data/ODI-2018.csv")
print("LENGTH")
print(len(df))
print("*********************************************************")

print("KEYS")
print(df.keys())
print("*********************************************************")

print("FIRST 5")
print(df.head(5))
print("*********************************************************")

print("UNIQUE VALUES")
for key in df.keys():
  print(key)
  print(list(df[key].unique()))
  print("")
print("*********************************************************")

# print("Value_counts")
# for key in df.keys():
#   print(df.groupby(key).count())
# print("*********************************************************")