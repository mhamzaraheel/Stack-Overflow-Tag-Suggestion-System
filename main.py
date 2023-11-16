# to read the file
import pandas as pd

# for calculation
import numpy as np

# for evaluation of literals
import ast

# to save the module
import pickle

# to neat the data
import neattext as nt
import neattext.functions as nfx

# for vectorization
from sklearn.feature_extraction.text import TfidfVectorizer
# for preprocessing of data
from sklearn.preprocessing import MultiLabelBinarizer

# to split the data
from sklearn.model_selection import train_test_split

# all classifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression

# for OneVsRestClassifier
from sklearn.multiclass import OneVsRestClassifier


# ----------------------------------------------------------------------------------------------------
# Load the data

data = pd.read_csv("data\stackoverflow.csv", index_col=0)
print(f"\n\nThere are {len(data)} rows in the dataset.\n")
print(data.head)
print(f"\nShape of Data is {data.shape}",)  
# print(data)

# -------------------------------------------------------------------------------------------------------
# to check the duplication in out text

total_duplicate_titles = sum(data["Text"].duplicated())
print(f"\nThere are {total_duplicate_titles} duplicate titles.")


# -----------------------------------------------------------------------------------------------

# There are some terms with occurrence as low as 1.
lowOccurence = sum(data["Tags"].value_counts() == 1)
print(f"\nData with occurrence as low as 1 = {lowOccurence}")   


# ------------------------------------------------------------------------------------------------


# Filtering the rare terms.(occurrence low as 1)
data = data.groupby("Tags").filter(lambda x: len(x) > 1)
# 176 data will be removed after this
# print(data.shape)


# --------------------------------------------------------------------------------------------

# How many unique Tags?
unique = data["Tags"].nunique()
print(f"\nUnique tags in file {unique}")

# ------------------------------------------------------------------------------------------------


# drop the tag where value is Null or empty

nullValue = data["Text"] != ""
data = data[nullValue]
data = data.dropna()

# print(data.shape)
# there is nothing Null


# ---------------------------------------------------------------------------------------------------------

# tags is of type string

print(data["Tags"].iloc[0])
print("First",type(data["Tags"].iloc[0]))

# -----------------------------------------------------------------------------------------------------------


# lambda function to apply literal-evaluation on all data
data["Tags"] = data["Tags"].apply(lambda x: ast.literal_eval(x))
# after this all tag column convert into list
print(type(data["Tags"].iloc[0]))
print(data["Tags"].iloc[0])


# ---------------------------------------------------------------------------------------------------------


# convert tag into onehot encoding
# print(data["Tags"])
multilabel = MultiLabelBinarizer()
y = multilabel.fit_transform(data["Tags"])
print(y)
print(f"\nClasses in the Data{multilabel.classes_}")


# ---------------------------------------------------------------------------------------------------------
# number of stopwords in every row

print("\nStop Words in every Row")
# stopWords = data["Text"].apply(lambda x:nt.TextFrame(x).count_stopwords())
stopWords = data["Text"].apply(lambda x:nt.TextExtractor(x).extract_stopwords())
print(stopWords)


# ---------------------------------------------------------------------------------------------------------
# with the help of neattext function we can do these following things'
# dir(nfx)

# we re going to remove te stop words
corpus = data["Text"].apply(nfx.remove_stopwords)

# ---------------------------------------------------------------------------


# tfidf vectorizer
tfidf = TfidfVectorizer(analyzer="word", max_features=10000, lowercase=True, encoding="utf-8",
                        stop_words="english")
x = tfidf.fit_transform(corpus)
# print(x.shape)


# features learn
# print(tfidf.vocabulary_)


# ---------------------------------------------------------------------------

# train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=115,stratify=data["Tags"].values)  # 42


# ---------------------------------------------------------------------------

# finding jcard score
# in multilabel ...performance measures something other than binary classification

def j_score(y_true, y_pred):
    jaccard = np.minimum(y_true, y_pred).sum(axis=1) / np.maximum(y_true, y_pred).sum(axis=1)
    return jaccard.mean() * 100


def print_Score(y_pred, clf):
    print("CLF:", clf.__class__.__name__)
    print("Jaccard Score: {}".format(j_score(y_test, y_pred)))
    print("--------------------")


# ---------------------------------------------------------------------------------------------
# Naive Byes

classifier = MultinomialNB()

# one vs rest approach
clf = OneVsRestClassifier(classifier)

# train the data
clf.fit(x_train, y_train)

# test the data
y_pred = clf.predict(x_test)

# printig the score
print_Score(y_pred, classifier)

# ---------------------------------------------------------------------------------------------

# Naive Byes

classifier = GaussianNB()


# one vs rest approach
clf = OneVsRestClassifier(classifier)

# train the data
clf.fit(x_train.toarray(), y_train)

# test the data
y_pred = clf.predict(x_test.toarray())

# printig the score
print_Score(y_pred, classifier)

# ---------------------------------------------------------------------------------------------

# KNN

classifier = KNeighborsClassifier(n_neighbors=10,metric="euclidean")

# one vs rest approach
clf = OneVsRestClassifier(classifier)

# train the data
clf.fit(x_train, y_train)

# test the data
y_pred = clf.predict(x_test)

# printing the score
print_Score(y_pred, classifier)

# -----------------------------------------------------------------------------------------------------


# Decision Tree
classifier = DecisionTreeClassifier(criterion="entropy", random_state=0, max_depth=20, min_samples_leaf=10)

# one vs rest approach

clf = OneVsRestClassifier(classifier)

# train the data
clf.fit(x_train, y_train)

# test the data
y_pred = clf.predict(x_test)

# printing the score
print_Score(y_pred, classifier)


# ----------------------------------------------------------------------------------------------

# SGDClassifier
classifier = SGDClassifier(max_iter=3000,shuffle=True,random_state=115)

# one vs rest approach
clf = OneVsRestClassifier(classifier)

# train the data
sgdClassifier = clf.fit(x_train, y_train)

# test the data
y_pred = clf.predict(x_test)

# printing the score
print_Score(y_pred, classifier)


with open("SGDClassifier.pkl", "wb") as file:
    pickle.dump(sgdClassifier, file)

# ------------------------------------------------------------------------------------------------------

# Logistic Regression

classifier = LogisticRegression(C=15, solver="lbfgs",max_iter=300,random_state=115)

# one vs rest approach
clf = OneVsRestClassifier(classifier)


# train the data
logisticRegression = clf.fit(x_train, y_train)


# test the data
y_pred = clf.predict(x_test)

# printing the score
print_Score(y_pred, classifier)

with open("LogisticRegression.pkl", "wb") as file:
    pickle.dump(logisticRegression, file)

# -------------------------------------------------------------------------------------------------

# SVC linear

classifier = LinearSVC(C=2, multi_class="crammer_singer", dual=True, random_state=115, max_iter=1000)

clf = OneVsRestClassifier(classifier)

# train the data
linearSVC = clf.fit(x_train, y_train)

# test the data
y_pred = clf.predict(x_test)

# printing the score
print_Score(y_pred, classifier)

# save the model
import pickle

with open("LinearSVC.pkl", "wb") as file:
    pickle.dump(linearSVC, file)

# -------------------------------------------------------------------------------------------------

# test the model on real world example

with open("LogisticRegression.pkl", "rb") as file:
    logisticRegression = pickle.load(file)




x = ["Is there any solution for this, I need to css from style tag into style, this code below works when I open it, "
     "but when I publish it doesn't work. I'm making website in web-flow so I need to put that code in HTML Embedand "
     "that's why "]
xt = tfidf.transform(x)


answer = multilabel.inverse_transform(logisticRegression.predict(xt))
print(answer)
