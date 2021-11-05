import json
from kafka import KafkaProducer
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import requests
from lxml import etree
from bs4 import BeautifulSoup
import time
import re
import urllib.parse

from weibo_top_sentiment import *

bayesSentimentModel = BayesSentiment()

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.54 Safari/537.36',
    'Cookie': 'SINAGLOBAL=8952596771843.973.1619269017487; SCF=AqdkznlYHewYaGz11U_Ewp7RO2xO2Xd4cvwl_zD0tRWF0Wm2BeSnOv3Y-wXnvZYRLf0ffP44xgWozS0YFaAM8eI.; ALF=1663466356; UOR=cn.bing.com,weibo.com,localhost:8888; SUB=_2A25MZGr7DeRhGeNI7VUV8y_IzTiIHXVvp3azrDV8PUJbkNAKLVjCkW1NSCumaJHCc-hZj7e5J3Q4wPOjB0zVojev; SUBP=0033WrSXqPxfM725Ws9jqgMF55529P9D9WFsS52AoM-aTTy1EieyUG8Z5NHD95QfSoqNShepShqXWs4Dqcj6i--RiKLFiKyFi--NiKnEiK.Ei--4i-2fi-82i--Xi-zRiKy2i--fiKysi-8si--fiKyWi-27i--RiKLFiKyF; _s_tentry=-; Apache=8470921214117.056.1634791170888; ULV=1634791170895:23:7:1:8470921214117.056.1634791170888:1634256856266'}


def get_top(producer):
    url = "https://s.weibo.com/top/summary"
    r = requests.get(url, headers=headers)
    # print(res.text)
    soup = BeautifulSoup(r.text, "html.parser")
    for i in soup.select('tbody tr'):
        numtmp = i.select('.ranktop')
        if len(numtmp):
            try:
                num = int(numtmp[0].string)
                title = i.select('.td-02 a')[0].string
                hot_score = i.select('.td-02 span')[0].string.split(' ')[-1]
                # yield num, key, hot
                print(num, title, 'hot:', hot_score)
                result = get_weibo_list("https://s.weibo.com//weibo?q=%23" + title[0] + "%23&Refer=top")
            except ValueError as e:
                continue
            for each in result:
                sentiment_score = bayesSentimentModel.predict(each)
                print(title, sentiment_score)
                msg = {"title": title, "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),"sentiment_score": sentiment_score}
                producer.send('weibotop', key='ff', value=json.dumps(msg))

        time.sleep(2)


def get_weibo_list(url):
    r = requests.get(url, headers=headers)

    bs = BeautifulSoup(r.text)
    body = bs.body
    div_m_main = body.find('div', attrs={'class': 'm-main'})
    div_m_flex1 = div_m_main.find('div', attrs={'class': 'woo-box-flex'})
    div_m_wrap = div_m_flex1.find('div', attrs={'class': 'woo-box-flex', 'id': 'pl_feed_main'})
    div_m_con_l = div_m_wrap.find('div', attrs={'class': 'main-full'})
    data_div = div_m_con_l.findAll('div', attrs={'class': 'card-wrap', 'action-type': 'feed_list_item'})

    weibo_list = []
    for each_div in data_div:
        div_card = each_div.find('div', attrs={'class': 'card'})
        div_card_feed = div_card.find('div', attrs={'class': 'card-feed'})
        div_content = div_card_feed.find('div', attrs={'class': 'content'})

        p_feed_list_content = div_content.find('p', attrs={'class': 'txt', 'node-type': 'feed_list_content'})
        content_text = p_feed_list_content.get_text()
        p_feed_list_content_full = div_content.find('p', attrs={'class': 'txt', 'node-type': 'feed_list_content_full'})
        if p_feed_list_content_full:
            content_text = p_feed_list_content_full.get_text()

        weibo_list.append(content_text.strip())

    return weibo_list


producer = KafkaProducer(bootstrap_servers='localhost:9092',
                         key_serializer=lambda k: json.dumps(k).encode(),
                         value_serializer=lambda v: json.dumps(v).encode())

while True:
    get_top(producer=producer)
    time.sleep(600)
