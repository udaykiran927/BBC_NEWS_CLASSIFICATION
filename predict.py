import pandas as pd
import numpy as np
import nltk
from nltk.stem import PorterStemmer
import pickle
from nltk.tokenize import word_tokenize 
#nltk.download('stopwords')
df=pd.read_csv("BBC_NEWS.csv")
df.drop("ArticleId",axis=1,inplace=True)
df['category_id'] = df['Category'].factorize()[0]
stopwords = nltk.corpus.stopwords.words('english')
df['news_without_stopwords'] = df['Text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))
ps = PorterStemmer()
df['news_porter_stemmed'] = df['news_without_stopwords'].apply(lambda x: ' '.join([ps.stem(word) for word in x.split()]))
df['news_porter_stemmed'] = df['news_porter_stemmed'].apply(lambda x: ' '.join(x.lower() for x in x.split()))
freq = pd.Series(' '.join(df['news_porter_stemmed']).split()).value_counts()
data = df[['Category', 'category_id', 'news_porter_stemmed']]
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2))
features = tfidf.fit_transform(data.news_porter_stemmed).toarray()
labels = data.category_id
data.columns = ['newstype', 'category_id', 'news_porter_stemmed']
category_id_df = data[['newstype', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'newstype']].values)
from sklearn.feature_selection import chi2
N = 5
for newstype, category_id in sorted(category_to_id.items()):
    features_chi2 = chi2(features, labels == category_id)
    indices = np.argsort(features_chi2[0])
    feature_names = np.array(tfidf.get_feature_names_out())[indices]
    unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
    bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
from sklearn.naive_bayes import ComplementNB as CNB
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(features, labels, test_size=0.25, random_state=0)
classifier=CNB()
classifier.fit(X_train,y_train)
pickle.dump(classifier,open("model.pkl","wb"))
