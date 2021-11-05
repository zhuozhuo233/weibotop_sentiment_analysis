from flask import *
from jinja2 import Markup
from pyecharts import options as opts
from pyecharts.charts import Bar
import random
import pandas as pd

app = Flask(__name__)

@app.route("/")
def index():
	bar = gen_data_bar()
	return Markup(bar.render_embed())

def gen_data_bar():
	df = pd.read_csv('weibo_sentiment_result.csv')
	title_names = []
	avg_scores = []
	df = df.sort_values(by="avg(sentiment_score)",ascending=False)
	for index in range(df.shape[0]):
		title_names.append(df['title'].iloc[index])
		avg_scores.append(str(round(float(df['avg(sentiment_score)'].iloc[index]),2)))

	bar = (Bar()
	.add_xaxis(title_names)
	.add_yaxis('得分',avg_scores)
	.set_global_opts(title_opts=opts.TitleOpts(title="微博情绪分析",subtitle=""),xaxis_opts=opts.AxisOpts(name="",axislabel_opts={"rotate":15},name_rotate=60))
	)
	
	return bar



if __name__=='__main__':
	app.run(host='192.168.91.30',port='8001',processes=1)
