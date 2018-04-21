from init import *
from functions import *

df = pd.read_csv("../Data/ODI-2018.csv")
# printInfo(df, "nope")
# vizBar(df)

# pie_array = []
# for key in df.keys():
#     if len([str(x).lower() for x in list(df[key].unique())]) < 10:
#         pie_array.append(key)
# vizPie(df, pie_array)
import re
bDay = [str(x).lower() for x in list(df["When is your birthday (date)?"])]
bDay2 = [x.replace("-", "/") for x in bDay]
bDay3 = [x.replace(".", "/") for x in bDay2]
bDay4 = [x.replace(" ", "/") for x in bDay3]
a = [re.split("/", x) for x in bDay4]
good = [x for x in a if len(x) == 3]
prachtig = []
for i in good:
    triple = []
    try:
        for j in range(len(i)):
            value = int(i[j])
            if value > 1918 and value < 2005 and j == 2:
                triple.append(value)
            elif j < 2:
                triple.append(value)
        if len(triple) == 3:
            prachtig.append(triple)
    except ValueError:
        continue
# print(good)
# print(prachtig)
ages = []
import datetime

def calculate_age(born):
    today = datetime.date.today()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))

for i in prachtig:
  try:
    birthday = str(i[0]) + " " + str(i[1]) + " " + str(i[2])
    ages.append(calculate_age(datetime.datetime.strptime(birthday, "%d %m %Y")))
  except ValueError:
    continue
print(len(ages))
