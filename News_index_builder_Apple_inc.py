#Here I combine the MSCI dataframe with the News Sentiment dataframe
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np

pd.set_option('display.max_rows', 1000)

def classifier(row):
	if row['deg_pos'] > 0.7:
		val = 1
	else:
		val = -1
	return val


df_news = pd.read_csv("D:/Python/Data/US_Financial_News/Apple_News_with_pred_Sentiment.csv", encoding='utf-8')
df_Apple = pd.read_csv("D:/Python/Data/Apple/AAPL.csv", thousands=',')

#df_news = df_news[~df_news['Site'].isin(['fortune.com'])]
df_news['pred_Sentiment'] = df_news.apply(classifier,axis=1)

df_sent = df_news.groupby('Date_Published')['pred_Sentiment'].count().reset_index(name="count")
df_sent['pos_minus_neg'] = df_news.groupby('Date_Published')['pred_Sentiment'].sum().reset_index(name="pos_minus_neg")["pos_minus_neg"]
#df_sent['pos_80_Change'] = np.log(df_sent.pred_Sentiment) - np.log(df_sent.pred_Sentiment.shift(1))
#df_sent['pos_count'] = df_news.groupby('Date_Published')['pred_Sentiment'].sum().reset_index(name="pos_count")["pos_count"]
#df_sent['pos_share'] = df_sent['pos_count'] / df_sent['count']
df_sent.rename(columns={'Date_Published': 'Date'}, inplace=True)
df_sent['Date'] =  pd.to_datetime(df_sent['Date'])
df_sent['Standardized_Index'] = df_sent['pos_minus_neg'] / df_sent['count']
df_sent['14_day_Moving_Average_index'] = (df_sent.Standardized_Index + df_sent.Standardized_Index.shift(1) + df_sent.Standardized_Index.shift(2)+ df_sent.Standardized_Index.shift(3)+ df_sent.Standardized_Index.shift(4)+ df_sent.Standardized_Index.shift(5)+ df_sent.Standardized_Index.shift(6)+ df_sent.Standardized_Index.shift(7)+ df_sent.Standardized_Index.shift(8)+ df_sent.Standardized_Index.shift(9)+ df_sent.Standardized_Index.shift(10)+ df_sent.Standardized_Index.shift(11)+ df_sent.Standardized_Index.shift(12)+ df_sent.Standardized_Index.shift(13)+ df_sent.Standardized_Index.shift(14))/15
print(list(df_sent.columns))

df_Apple['Date'] = df_Apple['Date'].apply(lambda x: dt.datetime.strptime(x,'%Y-%m-%d'))
df_Apple['Close'] = pd.to_numeric(df_Apple['Close'])
df_Apple['Change %'] = np.log(df_Apple.Close) - np.log(df_Apple.Close.shift(1))


df_combined = pd.merge(df_Apple, df_sent, on='Date', how='left')

df_index = df_combined[["Standardized_Index", "Date", "Close", "14_day_Moving_Average_index"]]
df_index.set_index('Date', inplace=True)
df_index.plot.line(subplots=True)
plt.show()

#WHICH NEWS SITES DID I USE
print(df_news["Site"].unique())
print(df_index[["Close","14_day_Moving_Average_index"]].corr())
# Testing the Neural Network on a sample from the news
#print(df_news.groupby(['Date_Published','Site'])['pred_Sentiment'].count().reset_index(name="count"))
#df_news[["Title","pred_Sentiment"]].sample(n=68,random_state=1).to_csv("D:/Python/Data/US_Financial_News/representative_sample_apple_sentiment.csv", encoding='utf-8', index=None)