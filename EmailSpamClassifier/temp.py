import numpy as np
import pandas as pd
df = pd.read_csv('spam.csv',encoding='latin-1')
df.sample(5)
df.shape
# Data cleaning
# EDA
# Text Preprocessing
# Model Building
# Evaluation
# Improvement
# Website
# Deploy
df.info()
df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)
df.sample(5)
df.rename(columns={'v1':'target','v2':'text'},inplace=True)
df.sample(5)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['target'] = le.fit_transform(df['target'])
df.isnull().sum()
df.duplicated().sum()
df.drop_duplicates(keep='first',inplace=True)
df.duplicated().sum()
df.shape
df.head()
df['target'].value_counts()
import matplotlib.pyplot as plt
plt.pie(df['target'].value_counts(),labels=['ham','spam'],autopct='%0.2f')
plt.show()
df['num_char'] = df['text'].apply(len)
df.head()
import nltk
df['num_words'] = df['text'].apply(lambda x: len(nltk.word_tokenize(x)))
df['num_sentence'] = df['text'].apply(lambda x: len(nltk.sent_tokenize(x)))
df.head()
df[['num_char','num_words','num_sentence']].describe()
df[df['target']==0][['num_char','num_words','num_sentence']].describe()
df[df['target']==1][['num_char','num_words','num_sentence']].describe()
import seaborn as sns
sns.histplot(df[df['target'] == 0]['num_char'])
sns.histplot(df[df['target'] == 1]['num_char'], color='red')
sns.pairplot(df,hue='target')
sns.heatmap(df[['target','num_char','num_words','num_sentence']].corr(),annot=True)
from nltk.corpus import stopwords
# stopwords.words('english')
from nltk.stem.porter import PorterStemmer
import string
def transform_text(text):
    ps = PorterStemmer()
    y = []
    text = text.lower()
    text = nltk.word_tokenize(text)
    for i in text:
        if i.isalnum():
            y.append(i)
    x = []
    for i in y:
        if i not in stopwords.words('english') and i not in string.punctuation:
            x.append(ps.stem(i))
            
    return x
transform_text("I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k? I've cried enough today.")
df['transformed_text'] = df['text'].apply(transform_text)
df.head()
df.head()
df[df['target']==1]['transformed_text'].tolist()[:3]
spam_corpus = []
for msg in df[df['target']==1]['transformed_text'].tolist():
    for word in msg:
        spam_corpus.append(word)
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
c = Counter(spam_corpus).most_common(30)
df_common = pd.DataFrame(c,columns=['word','count'])
sns.barplot(x='word', y='count', data=df_common)
plt.xticks(rotation=90)
plt.show()
len(spam_corpus)
ham_corpus = []
for msg in df[df['target']==0]['transformed_text'].tolist():
    for word in msg:
        ham_corpus.append(word)
c = Counter(ham_corpus).most_common(30)
df_common = pd.DataFrame(c,columns=['word','count'])
sns.barplot(x='word', y='count', data=df_common)
plt.xticks(rotation=90)
plt.show()
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
cv = CountVectorizer()
# 2
tfidf = TfidfVectorizer(max_features=3000)
# 1
# tfidf = TfidfVectorizer()
df['transformed_text'].head()
df['transformed_text'] = df['transformed_text'].apply(lambda x: " ".join(x))
df['transformed_text'].head()
X = cv.fit_transform(df['transformed_text']).toarray()
X_tfidf = tfidf.fit_transform(df['transformed_text']).toarray()
X.shape
X_tfidf.shape
y = df['target'].values
y
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2
)
X_traintfidf, X_testtfidf, y_traintfidf, y_testtfidf = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=2
)
X_traintfidf_n, X_testtfidf_n, y_traintfidf_n, y_testtfidf_n = train_test_split(
    X_n, y_n, test_size=0.2, random_state=2
)
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score
gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()

gnbtfidf = GaussianNB()
mnbtfidf = MultinomialNB()
bnbtfidf = BernoulliNB()
gnb.fit(X_train,y_train)
gnbtfidf.fit(X_traintfidf,y_traintfidf)
mnb.fit(X_train,y_train)
mnbtfidf.fit(X_traintfidf,y_traintfidf)
bnb.fit(X_train,y_train)
bnbtfidf.fit(X_traintfidf,y_traintfidf)
y_pred1 = gnb.predict(X_test)
print(accuracy_score(y_test, y_pred1))
print(confusion_matrix(y_test, y_pred1))
print(precision_score(y_test, y_pred1))
y_pred2 = gnbtfidf.predict(X_testtfidf)
print(accuracy_score(y_testtfidf, y_pred2))
print(confusion_matrix(y_testtfidf, y_pred2))
print(precision_score(y_testtfidf, y_pred2))
y_pred3 = mnb.predict(X_test)
print(accuracy_score(y_test, y_pred3))
print(confusion_matrix(y_test, y_pred3))
print(precision_score(y_test, y_pred3))
y_pred4 = mnbtfidf.predict(X_testtfidf)
print(accuracy_score(y_testtfidf, y_pred4))
print(confusion_matrix(y_testtfidf, y_pred4))
print(precision_score(y_testtfidf, y_pred4))
y_pred5 = bnb.predict(X_test)
print(accuracy_score(y_test, y_pred5))
print(confusion_matrix(y_test, y_pred5))
print(precision_score(y_test, y_pred5))
y_pred6 = bnbtfidf.predict(X_testtfidf)
print(accuracy_score(y_testtfidf, y_pred6))
print(confusion_matrix(y_testtfidf, y_pred6))
print(precision_score(y_testtfidf, y_pred6))
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
svc = SVC(kernel='sigmoid', gamma=1.0)
knc = KNeighborsClassifier()
mnb = MultinomialNB()
dtc = DecisionTreeClassifier(max_depth=5)
lrc = LogisticRegression(solver='liblinear', penalty='l1')
rfc = RandomForestClassifier(n_estimators=50, random_state=2)
abc = AdaBoostClassifier(n_estimators=50, random_state=2)
bc = BaggingClassifier(n_estimators=50, random_state=2)
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)
gbdt = GradientBoostingClassifier(n_estimators=50,random_state=2)
xgb = XGBClassifier(n_estimators=50,random_state=2)
clfs = {
    'SVC' : svc,
    'KN' : knc, 
    'NB': mnb, 
    'DT': dtc, 
    'LR': lrc, 
    'RF': rfc, 
    'AdaBoost': abc, 
    'BgC': bc, 
    'ETC': etc,
    'GBDT':gbdt,
    'xgb':xgb
}
def train_classifier(clf,X_train,y_train,X_test,y_test):
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    
    return accuracy,precision
train_classifier(svc,X_train,y_train,X_test,y_test)
accuracy_scores = []
precision_scores = []

for name,clf in clfs.items():
    
    current_accuracy,current_precision = train_classifier(clf, X_train,y_train,X_test,y_test)
    
    print("For ",name)
    print("Accuracy - ",current_accuracy)
    print("Precision - ",current_precision)
    
    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)
performance_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy':accuracy_scores,'Precision':precision_scores}).sort_values('Precision',ascending=False)
performance_df
performance_df1 = pd.melt(performance_df, id_vars = "Algorithm")
performance_df1
sns.catplot(x = 'Algorithm', y='value', 
               hue = 'variable',data=performance_df1, kind='bar',height=5)
plt.ylim(0.5,1.0)
plt.xticks(rotation='vertical')
plt.show()
# before applying Max_iter in tfidf vectorizor
accuracy_scores = []
precision_scores = []

for name,clf in clfs.items():
    
    current_accuracy,current_precision = train_classifier(clf, X_traintfidf,y_traintfidf,X_testtfidf,y_testtfidf)
    
    print("For ",name)
    print("Accuracy - ",current_accuracy)
    print("Precision - ",current_precision)
    
    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)
temp_df = pd.DataFrame({
    'Algorithm': clfs.keys(),
    'Accuracy_tfidf': accuracy_scores,
    'Precision_tfidf': precision_scores
})
# 2. Merge them together side-by-side using the Algorithm name!
performance_df = performance_df.merge(temp_df, on='Algorithm')
display(performance_df)
# After setting max_iter = 3000 to tfidf
accuracy_scores = []
precision_scores = []

for name,clf in clfs.items():
    
    current_accuracy,current_precision = train_classifier(clf, X_traintfidf,y_traintfidf,X_testtfidf,y_testtfidf)
    
    print("For ",name)
    print("Accuracy - ",current_accuracy)
    print("Precision - ",current_precision)
    
    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)
temp_df = pd.DataFrame({
    'Algorithm': clfs.keys(),
    'Accuracy_tfidf_max_iter=3000': accuracy_scores,
    'Precision_tfidf_max_iter=3000': precision_scores
})
# 2. Merge them together side-by-side using the Algorithm name!
performance_df = performance_df.merge(temp_df, on='Algorithm')
display(performance_df)
performance_df
tfidf = TfidfVectorizer(max_features=3000)
# After Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_n = tfidf.fit_transform(df['transformed_text']).toarray()
X_n = scaler.fit_transform(X_n)
y_n = df['target'].values
X_traintfidf, X_testtfidf, y_traintfidf, y_testtfidf = train_test_split(
    X_n, y_n, test_size=0.2, random_state=2
)
accuracy_scores = []
precision_scores = []

for name,clf in clfs.items():
    
    current_accuracy,current_precision = train_classifier(clf, X_traintfidf,y_traintfidf,X_testtfidf,y_testtfidf)
    
    print("For ",name)
    print("Accuracy - ",current_accuracy)
    print("Precision - ",current_precision)
    
    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)
temp_df = pd.DataFrame({
    'Algorithm': clfs.keys(),
    'Accuracy_tfidf_scaled': accuracy_scores,
    'Precision_tfidf_scaled': precision_scores
})
# 2. Merge them together side-by-side using the Algorithm name!
performance_df = performance_df.merge(temp_df, on='Algorithm')
display(performance_df)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np

cv = CountVectorizer()
tfidf = TfidfVectorizer(max_features=3000)


X = tfidf.fit_transform(df['transformed_text']).toarray()

X = np.hstack((X, df['num_char'].values.reshape(-1, 1)))
X.shape
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

# Scale only last column
X[:, -1] = scaler.fit_transform(X[:, -1].reshape(-1, 1)).ravel()
y = df['target'].values
y
X_traintfidf, X_testtfidf, y_traintfidf, y_testtfidf = train_test_split(
    X, y, test_size=0.2, random_state=2,stratify=y
)
X_traintfidf.shape
X_traintfidf
clfs = {
    'SVC' : svc,
    'KN' : knc, 
    'NB': mnb, 
    'DT': dtc, 
    'LR': lrc, 
    'RF': rfc, 
    'AdaBoost': abc, 
    'BgC': bc, 
    'ETC': etc,
    'GBDT':gbdt,
    'xgb':xgb
}
accuracy_scores = []
precision_scores = []

for name,clf in clfs.items():
    
    current_accuracy,current_precision = train_classifier(clf, X_traintfidf,y_traintfidf,X_testtfidf,y_testtfidf)
    
    print("For ",name)
    print("Accuracy - ",current_accuracy)
    print("Precision - ",current_precision)
    
    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)
temp_df = pd.DataFrame({
    'Algorithm': clfs.keys(),
    'Accuracy_tfidf_num_char_scaled': accuracy_scores,
    'Precision_tfidf_num_char_scaled': precision_scores
})
# 2. Merge them together side-by-side using the Algorithm name!
performance_df = performance_df.merge(temp_df, on='Algorithm')
display(performance_df)
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier

svc = SVC(probability=True, class_weight='balanced')
mnb = MultinomialNB()
etc = ExtraTreesClassifier(n_estimators=100, random_state=2)

voting = VotingClassifier(
    estimators=[('svm', svc), ('nb', mnb), ('et', etc)],
    voting='soft'
)

# Fit the model on training data
voting.fit(X_train, y_train)

# Predict on test data
y_pred = voting.predict(X_test)

# Evaluate performance
print("Accuracy", accuracy_score(y_test, y_pred))
print("Precision", precision_score(y_test, y_pred))
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, precision_score
from sklearn.svm import SVC

svc = SVC(probability=True)

# Define the voting classifier with three base models
voting = VotingClassifier(
    estimators=[('svm', svc), ('nb', mnb), ('et', etc)],
    voting='soft'
)

# Fit the model on training data
voting.fit(X_train, y_train)

# Predict on test data
y_pred = voting.predict(X_test)

# Evaluate performance
print("Accuracy", accuracy_score(y_test, y_pred))
print("Precision", precision_score(y_test, y_pred))
performance_df.to_csv('performance.csv')
# Clearly   (Accuracy_tfidf_num_char_scaled	Precision_tfidf_num_char_scaled     NB)      Performed the Best
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
mnb.fit(X_traintfidf,y_traintfidf)
y_pred = mnb.predict(X_testtfidf)
print("Accuracy", accuracy_score(y_testtfidf, y_pred))
print("Precision", precision_score(y_testtfidf, y_pred))

import pickle
pickle.dump(tfidf,open('vectorizer.pkl','wb'))
pickle.dump(mnb,open('model.pkl','wb'))
print('Done ')
