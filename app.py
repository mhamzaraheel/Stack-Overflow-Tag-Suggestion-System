# important libraries

# for GUI
from flask import Flask, render_template, request
# for loading the module
import pickle
# for literals evaluation
import ast
# to read th csb file
import pandas as pd
# to neat the data
import neattext.functions as nfx
# for tokenization
from sklearn.feature_extraction.text import TfidfVectorizer
# for preprocessing of label tags
from sklearn.preprocessing import MultiLabelBinarizer

# read the file
data = pd.read_csv("data\stackoverflow.csv", index_col=0)
# Filtering the rare terms.(occurrence low as 1)
data = data.groupby("Tags").filter(lambda x: len(x) > 1)
# literal evaluation(convert "Tags" into array)
data["Tags"] = data["Tags"].apply(lambda x: ast.literal_eval(x))
#  convert tag into onehot encoding
multilabel = MultiLabelBinarizer()
y = multilabel.fit_transform(data["Tags"])
# we re going to remove te stop words
corpus = data["Text"].apply(nfx.remove_stopwords)
# tfidf vectorizer
tfidf = TfidfVectorizer(analyzer="word", max_features=10000, ngram_range=(1, 1), lowercase=True, encoding="utf-8",stop_words="english")
tfidf.fit_transform(corpus)


# load the models
model_directory =  "trained_models"
with open(f"{model_directory}/SGDClassifier.pkl", "rb") as file:
    sgdClassifier = pickle.load(file)

with open(f"{model_directory}/LogisticRegression.pkl", "rb") as file:
    logisticRegression = pickle.load(file)

with open(f"{model_directory}/LinearSVC.pkl", "rb") as file:
    linearSVC = pickle.load(file)

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('Home.html')


@app.route('/predict', methods=['POST'])
def predict():

    inputs = []
    inputs.append(request.form["loginUser"])

    xt= tfidf.transform(inputs)

    prediction1 = multilabel.inverse_transform(sgdClassifier.predict(xt))
    prediction2 = multilabel.inverse_transform(logisticRegression.predict(xt))
    prediction3 = multilabel.inverse_transform(linearSVC.predict(xt))
    prediction1 = (str(prediction1).replace(' [', '').replace('[', '').replace(']', ''))
    prediction2 = (str(prediction2).replace(' [', '').replace('[', '').replace(']', ''))
    prediction3 = (str(prediction3).replace(' [', '').replace('[', '').replace(']', ''))

 

    return render_template('Home.html',sgd_result=prediction1,logistic_result = prediction2,svc_result = prediction3)


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)

