#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[18]:


import requests
import json
from lxml import etree
from bs4 import BeautifulSoup

headers={
    'Cookie':'SINAGLOBAL=5067277944781.756.1539012379187; SUBP=0033WrSXqPxfM725Ws9jqgMF55529P9D9WFXAgGhhYE-bL1UlBNsI6xh5JpX5KMhUgL.Foqce0eN1h2cehB2dJLoIEXLxK-L1h5LB-eLxK-L1h5LB-eLxK-L1K5L1heLxKBLBonL1h.LxKMLBKzL1KMt; UOR=jx3.xoyo.com,widget.weibo.com,login.sina.com.cn; ALF=1665190926; SSOLoginState=1633654926; SCF=AjnY75MXDIg2Sev-TVKQBdyuwLa-mrIYwFgLkjivnwGqe4HMR8MVkSqyfw315Fic7gc1c38G1W-RUtxrwPqe0qY.; SUB=_2A25MW-jeDeRhGeBI6FEW-C_KyziIHXVvEV0WrDV8PUNbmtAKLUzhkW9NRppHJg76K77LtSOxPlpC13YygxcK3EKM; _s_tentry=login.sina.com.cn; Apache=441836365226.03375.1633654927612; ULV=1633654927618:48:1:1:441836365226.03375.1633654927612:1632876696485',
    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.71 Safari/537.36'
}

def get_top():
    url = "https://s.weibo.com/top/summary"
    r = requests.get(url,headers=headers)
#     print(r.text)
#     print(r.status_code)
    html_xpath = etree.HTML(r.text)
    data = html_xpath.xpath('//*[@id="pl_top_realtimehot"]/table/tbody/tr[1]/td[2]')
    num = 1
    for tr in (data):
        print('-------------')
        title = tr.xpath('./a/text()')
        hot_score = tr.xpath('./span/text()')
        href = tr.xpath('./a/@href')
        if hot_score:
            print('{} {} hot: {}'.format(num,title[0],hot_score[0]))
            request = get_weibo_list('https://s.weibo.com/weibo?q=%23'+tittle[0]+'%23&Refer=top')
            print(result)
            num += 1
            
def get_weibo_list(url):
    r = requests.get(url,headers=headers)
    
    bs = BeautifulSoup(r.text)
    body = bs.body
    div_m_main = body.find('div',attrs={'class':'m-main'})
    div_m_wrap = div_m_main.find('div',attrs={'class':'m-wrap'})
    div_m_con_l = div_m_wrap.find('div',attrs={'class':'m-con-l'})
    data_div = div_m_con_l.findAll('div',attrs={'class':'card-wrap','action-type':'feed_list_item'})
    
    weibo_list = []
    for each_div in data_div:
        div_card = each_div.find('div',attrs={'class':'card'})
        div_card_feed = div_card.find('div',attrs={'class':'card-feed'})
        div_content = div_card_feed.find('div',attrs={'class':'content'})
        
        p_feed_list_content = div_content.find('p',attrs={'class':'txt','node-type':'feed_list_content'})
        content_text = p_feed_list_content.get_text()
        
        p_feed_list_content_full = div_content.find('p',attrs={'class':'txt','node-type':'feed_list_content_full'})
        if p_feed_list_content_full:
            content_text = p_feed_list_content_full.get_text()
            
        weibo_list.append(content_text.strip())
        
    return weibo_list
            
            


# In[19]:


get_top()


# In[20]:


#爬取“吴磊绝杀”标题的额主要内容
cont = get_weibo_list('https://s.weibo.com/weibo?q=%23%E6%AD%A6%E7%A3%8A%E7%BB%9D%E6%9D%80%23&Refer=top')
cont


# In[29]:


import re
import jieba
#去除噪声函数，去除微博内容中的特殊符号，语气助词等噪声、
def process(text):
    #去除url
    text = re.sub("(https?|ftp|file)://[-A-Za-z0-9+&@#/%=~_|]"," ",text)
    #去除@xxx(用户名)
    text = re.sub("@.+?( |$)", " ", text)
    #去除{%xxx%}（地理定位，微博话题等）
    text = re.sub("\{%.+?%\}", " ",text)
    #去除#xx#（标题引用）
    text = re.sub("\{#.+?#\}", " ", text)
    #去除【xx】(里面的内容通常都不是用户自己写的)
    text = re.sub("【.+?】", " ", text)
    #数据集中的噪声
    text = re.sub('\u200b'," ",text)
                  
    #分词
    words = [w for w in jieba.lcut(text) if w.isalpha()]
                
    result = " ".join(words)
    return result
                  


# In[30]:


#调用去噪函数处理爬取下来的内容
pro_cont = []
for each in cont:
    pro_cont.append(process(each))
pro_cont


# In[31]:


#为构建文本向量做准备，先转换成pd的DataFrame格式
import pandas as pd
df_title = pd.DataFrame(pro_cont,columns=['words'])
df_title.head(5)


# In[35]:


#定义训练样本数据，训练样本数据为用逗号分隔开的，字段含义为：id,情绪值，微博内容
#train.txt为训练数据 test.txt为测试数据

#加载停用词
stopwords = []
with open('stopwords.txt','r',encoding='utf-8') as f:
    for w in f:
        stopwords.append(w.strip())  #去除多余空格
stopwords  #查看停用词


# In[38]:


# 构建朴素贝叶斯分类模型
# 定义训练样本数据，训练样本数据为用逗号隔开的，字段含义为：id，情绪值，微博内容
# train.txt为训练数据，test.txt为测试数据
# 加载训练文本数据集、测试文本数据集的函数
# 训练文本数据集和测试文本数据集都为已打好标签的数据 1为正面情绪，0为负面情绪


# In[37]:


#加载数据，提取情绪分类和内容，进行处理
def load_corpus(path):
    data = []
    with open(path,'r',encoding='utf8') as f:
        for line in f:
            [_,sentiment,content] = line.split(',',2) #分隔三次
            #对微博内容调用process函数进行去噪处理
            content = process(content)
            data.append((content,int(sentiment)))
    return data


# In[39]:


#调用加载数据函数，用DataFrame的形式呈现
import pandas as pd
train_data = load_corpus('train.txt')
test_data = load_corpus('test.txt')

df_train = pd.DataFrame(train_data,columns=["words","label"])
df_test = pd.DataFrame(test_data,columns=["words","label"])
df_train.head(2)


# In[40]:


#用BOW 构建训练数据的文本向量
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(token_pattern='\[?\w+\]?',stop_words=stopwords)
X_train = vectorizer.fit_transform(df_train["words"])
y_train = df_train["label"]
print(type(X_train),X_train.shape)


# In[44]:


#用BOW构建测试数据的文本向量
X_test = vectorizer.transform(df_test["words"])
y_test = df_test["label"]


# In[45]:


#导入多项式朴素贝叶斯
from sklearn.naive_bayes import MultinomialNB
#实例化
clf = MultinomialNB()
#用训练模型训练数据
clf.fit(X_train,y_train)


# In[46]:


#在测试集上用模型预测结果
y_pred = clf.predict(X_test)
print(y_pred)


# In[47]:


#测试集效果检验
from sklearn import metrics

print(metrics.classification_report(y_test,y_pred))
print("准确率:",metrics.accuracy_score(y_test,y_pred))
auc_score = metrics.roc_auc_score(y_test,y_pred)  #先计算AUC
print("AUC:",auc_score)


# In[52]:


#对标题“吴磊绝杀”的微博主要内容运用模型预测情绪
#用BOW方式准备文本向量
x = vectorizer.transform(df_title['words'])
#预测
y_title = clf.predict(x)
#print(y_pred)
#导入numpy方便计算平均得分
import numpy as np
title = "吴磊绝杀"
print(title,'情感平均得分为：',np.mean(y_title))


# In[53]:


#保存模型
import pickle
with open('./mnb_model.pkl','wb') as f:
    pickle.dump(clf,f)


# In[54]:


#读取模型，预测结果
with open('./mnb_model.pkl','rb') as f:
    save_clf = pickle.load(f)
    
#读取的模型,预测结果
save_clf.predict(x)
import numpy as np
title = "吴磊绝杀"
print(title,'情感平均得分为:',np.mean(y_title))


# In[ ]:




