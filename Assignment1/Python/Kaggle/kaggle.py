from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import re


ROOT = os.path.abspath('../../Data/Kaggle')
TRAIN = os.path.join(ROOT, 'train.csv')
TEST = os.path.join(ROOT, 'test.csv')

train_set = pd.read_csv(TRAIN)


def get_age(data):
    return data[data.Age.notnull()].Age


def get_pclass(data):
    counts = data.Pclass.value_counts()
    return [counts[i+1] for i in range(3)]


def demographics():
    survivors = train_set[train_set.Survived == 1]
    victims = train_set[train_set.Survived == 0]

    male_survivors = survivors[survivors.Sex == 'male']
    female_survivors = survivors[survivors.Sex == 'female']

    male_victims = victims[victims.Sex == 'male']
    female_victims = victims[victims.Sex == 'female']

    # Age - Sex
    N = 10

    figure, axes = plt.subplots(2)

    axes[0].set_title("Male Survival over Age")
    axes[0].hist([get_age(male_survivors), get_age(male_victims)], N, histtype='barstacked', label=['Survivors', 'Victims'])
    axes[0].legend()

    axes[1].set_title("Female Survival over Age")
    axes[1].hist([get_age(female_survivors), get_age(female_victims)], N, histtype='barstacked', label=['Survivors', 'Victims'])
    axes[1].legend()

    axes[0].set_xlabel("age")
    axes[0].set_ylabel("frequency")
    axes[1].set_xlabel("age")
    axes[1].set_ylabel("frequency")



    plt.tight_layout()
    plt.show()

    # Sex - Class
    figure, axes = plt.subplots(2)

    axes[0].set_title("Male Survival over Class")
    axes[0].bar(np.arange(3), get_pclass(male_survivors), label='Survivors')
    axes[0].bar(np.arange(3), get_pclass(male_victims), bottom=get_pclass(male_survivors), label='Victims')
    axes[0].set_xticks(np.arange(3))
    axes[0].set_xticklabels([1, 2, 3])
    axes[0].set_xlabel("class")
    axes[0].set_ylabel("frequency")
    axes[0].legend()

    axes[1].set_title("Female Survival over Class")
    axes[1].bar(np.arange(3), get_pclass(female_survivors), label='Survivors')
    axes[1].bar(np.arange(3), get_pclass(female_victims), bottom=get_pclass(female_survivors), label='Victims')
    axes[1].set_xticks(np.arange(3))
    axes[1].set_xticklabels([1, 2, 3])
    axes[1].set_xlabel("class")
    axes[1].set_ylabel("frequency")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def prepare_features(data):
    age = np.array(data.Age)
    age_mask = ~np.isnan(age)
    age[~age_mask] = -1
    # age[np.logical_and(age >= 0, age < 18)] = 0
    # age[np.logical_and(age >= 18, age < 30)] = 1
    # age[np.logical_and(age >= 30, age <= 65)] = 2
    # age[age >= 65] = 1

    sex = np.array(data.Sex)
    sex[sex == 'male'] = 0
    sex[sex == 'female'] = 1

    cls = np.array(data.Pclass)

    sib_sp = np.array(data.SibSp)
    par_ch = np.array(data.Parch)

    family_size = sib_sp + par_ch + 1

    names = data.Name.str.split('[,.] ')
    titles = [name[1] for name in names.values]

    for i, title in enumerate(titles):
        if title in ("Mlle", "Mme"):
            titles[i] = "Mme"
        elif title == "Ms":
            titles[i] = "Miss"
        elif title in ("Don", "Sir", "Jonkheer", "Rev", "Major", "Col", "Capt", "Lady", "the Countess"):
            titles[i] = "Rare"

    titles = LabelEncoder().fit_transform(titles)

    family_names = [name[0] for name in names.values]
    family_names_unique = list(set(family_names))
    family_names_count = [family_names.count(name) for name in family_names_unique]
    big_family_names_unique = {name for name, count in zip(family_names_unique, family_names_count) if count > 2}
    family_names = [name if name in big_family_names_unique else 'small' for name in family_names]
    family_names = LabelEncoder().fit_transform(family_names)

    deck = LabelEncoder().fit_transform([c[0] if isinstance(c, str) else 'Unknown' for c in data.Cabin])

    room = [int(re.findall('(\d+)', c)[0]) if isinstance(c, str) and re.findall('(\d+)', c) else -1 for c in data.Cabin]

    fare = np.array(data.Fare)
    fare[np.isnan(fare)] = np.median(fare[~np.isnan(fare)])

    embark = np.array(data.Embarked)
    embark[embark == "C"] = 0
    embark[embark == "Q"] = 1
    embark[embark == "S"] = 2
    embark = embark.astype(np.float32)
    embark[np.isnan(embark)] = -1

    return np.array([sex, titles, fare, cls, room, family_size], np.float32).T

from sklearn import tree, ensemble, svm
from sklearn.model_selection import cross_val_score

data = prepare_features(train_set)
labels = train_set.Survived


clf = tree.DecisionTreeClassifier(max_depth=5)
scores = cross_val_score(clf, data, labels, cv=4)
print("Decision Tree          : Accuracy: {:3.5f} (+/- {:3.2f})".format(scores.mean(), scores.std() * 2))

# clf = ensemble.RandomForestClassifier(1000, max_depth=6, min_samples_leaf=0.01, min_samples_split=0.03, class_weight='balanced')
# scores = cross_val_score(clf, data, labels, cv=5)
# print("Random Forest          : Accuracy: {:3.5f} (+/- {:3.2f})".format(scores.mean(), scores.std() * 2))

clf.fit(data, labels)
test_set = pd.read_csv(TEST)
prediction = clf.predict(prepare_features(test_set))

submission = pd.DataFrame({"PassengerId": test_set["PassengerId"], "Survived": prediction})
submission.to_csv(os.path.join(ROOT, 'submission.csv'), index=False)

# predictors = [
#     "Sex", "Title", "Fare", "Class", "Family Size",
# ]
#
# clf.fit(data, labels)
#
# importances = np.array(clf.feature_importances_)
# importances_sorted = np.argsort(importances)
#
# plt.barh(np.arange(len(predictors)), importances[importances_sorted])
# plt.yticks(np.arange(len(predictors)), [predictors[i] for i in importances_sorted])
# plt.ylabel("Features")
# plt.xlabel("Importance")
# plt.tight_layout()
# plt.show()