import re 
import jieba
import sys
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import pickle

def load_corpus(path):
    data = []
    with open(path,'r',encoding='utf-8')as f:
        for line in f :
            [_,sentiment,content] = line.split(",",2)
            content = process(content)
            data.append((content,int(sentiment)))
    return data

def process(text):
    text = re.sub("(http?|ftp|file)://[-A-Za-z0-9+%@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#?%=~|]"," ",text)
    text = re.sub("@.+?( |$)"," ",text)
    text = re.sub("\{%.+?%\}"," ",text)
    text = re.sub("\{#.+?%\}"," ",text)
    text = re.sub("【.+?】"," ",text)
    text = re.sub("\u200b"," ",text)
    text = re.sub("n"," ",text)
    words = [w for w in jieba.lcut(text) if w.isalpha()]
    result =" ".join(words)
    return result

TRAIN_PATH = "/opt/train.txt"
TEST_PATH = "/opt/test.txt"


def train():
    stopwords = []
    with open('/opt/stopwords.txt','r',encoding='utf-8')as f:
        for w in f :
            stopwords.append(w.strip())
    
    train_data = load_corpus(TRAIN_PATH)
    test_data = load_corpus(TEST_PATH)
    df_train = pd.DataFrame(train_data,columns=["words","label"])
    df_test = pd.DataFrame(test_data,columns=["words","label"])
    vectorizer = CountVectorizer(token_pattern='\[?\w+\]?',stop_words=stopwords)
    x_train = vectorizer.fit_transform(df_train['words'])
    y_train = df_train["label"]
    x_test = vectorizer.transform(df_test["words"])
    y_test = df_test["label"]
    clf = MultinomialNB()
    clf.fit(x_train,y_train)
    y_pred =clf.predict(x_test)
    
    print(metrics.classification_report(y_test,y_pred))
    print("准确率:",metrics.accuracy_score(y_test,y_pred))
    with open('bayes_model.pkl','wb')as f:
        pickle.dump([clf,vectorizer],f)

class BayesSentiment(object):
    def __init__(self):
        with open('bayes_model.pkl','rb') as f:
            self.clf,self.vectorizer = pickle.load(f)
    def predict(self,sentence):
        sentenceprocessed = [process(sentence)]
        vec = self.vectorizer.transform(sentenceprocessed)
        return self.clf.predict_proba(vec)[0][1]
    
if __name__ == '__main__':
    train()
