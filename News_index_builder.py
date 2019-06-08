#Here I combine the MSCI dataframe with the News Sentiment dataframe
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import numpy as np

def classifier(row):
	if row['deg_pos'] > 0.7:
		val = 1
	else:
		val = -1

	return val


df_news = pd.read_csv("D:/Python/Data/US_Financial_News/News_with_pred_Sentiment.csv", encoding='utf-8')
df_MSCI = pd.read_csv("D:/Python/Data/MSCI_US_index/MSCI USA Historical Data.csv", thousands=',')

df_news['pred_Sentiment_80'] = df_news.apply(classifier,axis=1)

df_sent = df_news.groupby('Date_Published')['pred_Sentiment'].count().reset_index(name="count")
df_sent['pos_minus_neg'] = df_news.groupby('Date_Published')['pred_Sentiment_80'].sum().reset_index(name="pos_minus_neg")["pos_minus_neg"]
df_sent['pos_80_Change'] = np.log(df_sent.pos_minus_neg) - np.log(df_sent.pos_minus_neg.shift(1))
df_sent['pos_count'] = df_news.groupby('Date_Published')['pred_Sentiment'].sum().reset_index(name="pos_count")["pos_count"]
df_sent['pos_share'] = df_sent['pos_count'] / df_sent['count']
df_sent.rename(columns={'Date_Published': 'Date'}, inplace=True)
df_sent['Date'] =  pd.to_datetime(df_sent['Date'])
df_sent['Standardized_Index'] = df_sent['pos_minus_neg'] / df_sent['count']

df_MSCI['Date'] = df_MSCI['Date'].apply(lambda x: dt.datetime.strptime(x,'%b %d, %Y'))
df_MSCI['Price'] = pd.to_numeric(df_MSCI['Price'])
df_MSCI['Change %'] = np.log(df_MSCI.Price) - np.log(df_MSCI.Price.shift(1))


df_combined = pd.merge(df_MSCI, df_sent, on='Date', how='left')

df_for_plot = df_combined[["Price","pos_minus_neg","Date", "count"]]
df_for_plot.set_index('Date', inplace=True)

df_index = df_combined[["Standardized_Index", "Date"]]
df_index.set_index('Date', inplace=True)
df_index.plot.line()
plt.show()

print(df_for_plot)
df_for_plot.plot.line()
plt.show()