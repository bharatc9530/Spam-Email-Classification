from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS
import joblib

NB_spam_model = open('NB_spam_model.pkl','rb')
clf = joblib.load(NB_spam_model)
app = Flask(__name__)



@app.route('/')
def man():
	return render_template('home.html')


@app.route('/predict', methods=['POST'])
def home():
	za = pd.read_csv("datasets_483_982_spam.csv",encoding="latin-1")
	za.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True, axis=1)
	za.columns = ['label','message']
	za['label'] = za['label'].replace({'ham':0,'spam':1}, inplace=True)
	def text_process(mess):
		STOPWORDS = stopwords.words('english') + ['u', 'ü', 'ur', '4', '2', 'im', 'dont', 'doin', 'ure']
		nopunc = [char for char in mess if char not in string.punctuation]
		nopunc = ''.join(nopunc)
		return ' '.join([word for word in nopunc.split() if word.lower() not in STOPWORDS])
	textFeatures = za['message'].copy()
	textFeatures = textFeatures.apply(text_process)
	vectorizer = TfidfVectorizer("english")
	l = vectorizer.fit_transform(textFeatures)

	data1 = request.form['a']
	data2 = [data1]
	features = vectorizer.transform(data2).toarray()
	pred = clf.predict(features)
	return render_template('after.html', data=pred)


if __name__ == "__main__":
    app.run(debug=True)















