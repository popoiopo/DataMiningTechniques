from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import numpy as np
import os

PATH = os.path.abspath('../../Data/SmsCollection.csv')

x = []
y = []


# Reading And Transforming Data #
with open(PATH) as label_txt_file:
    for entry in label_txt_file:
        label, txt = entry.split(';', 1)
        x.append(txt)
        y.append(1 if label == 'spam' else 0)

# True if 'spam', False if 'ham' #
y = np.array(y, np.bool)

import matplotlib.pyplot as plt
from wordcloud import WordCloud
wc = WordCloud(width=1024, height=768, background_color='white').generate(' '.join([x_ for x_, y_ in zip(x, y) if y_ == True]))
plt.imshow(wc)
plt.axis('off')
plt.tight_layout()
plt.show()

# Word Counts #
vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w\w\w+\b")
x = vectorizer.fit_transform(x)

mask = np.random.rand(len(y)) < 0.5

x_train = x[mask]
y_train = y[mask]

x_test = x[~mask]
y_test = y[~mask]

classifier = MultinomialNB()
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)

print(metrics.classification_report(y_test, y_pred))

score = cross_val_score(classifier, x, y, cv=10)
print("Accuracy    : {:3.5f} (+/- {:3.2f})".format(np.mean(score), np.std(score)))

print()
print("Type your own spam/ham SMS!")
print()
while True:
    prediction = classifier.predict(vectorizer.transform([input("SMS: ")]))
    print("This SMS is labeled as {}!\n".format('SPAM' if prediction else 'ham'))