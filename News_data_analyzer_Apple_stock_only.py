import pandas as pd
import datetime as dt
import matplotlib
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', 1201)

df = pd.read_csv("D:/Python/Data/US_Financial_News/2018_01_05.csv", encoding='utf-8')
print(df.dtypes)
export_data = "D:/Python/Data/US_Financial_News/Apple_News_2018_01_05.csv"

#print(df['Section_title'].nunique())
#print(df['Section_title'].value_counts())
##########################################################
#with open("MyFile.txt","w", encoding="utf-8") as file:
#	file.write(str(df['Section_title'].value_counts()))
##########################################################

#To line 172 count: 228
#list_of_sectiontitles = ['Press Releases - CNBC','Reuters: Company News','Reuters: Business News', 'Top News and Analysis (pro)',
#						'The Wall Street Journal &amp; Breaking News Business Financial and Economic News World News and Video', 
#						'Business &amp; Financial News U.S &amp; International Breaking News | Reuters', 'WSJ.com: US Business',
#						'Reuters: U.S.', 'Markets &amp; Finance News | Reuters.com', 'Reuters: Politics', 'Stock Market &amp; Finance News - Wall Street Journal',
#						'Stock Picks', 'US Stock Market News - Dow Jones Nasdaq S&amp;P 500 | Reuters.com', 'US News Breaking News and Headlines - Wall Street Journal',
#						'Reuters: Stocks & Shares News', 'Business &amp; Finance News - Wall Street Journal']


searchfor = ["Apple", "AAPL", "Apple's"]
#df2 = df[df['Section_title'].isin(list_of_sectiontitles)]
df2 = df[df['Title'].str.contains('|'.join(searchfor))==True]
df2['Date_Published'] = df2['Date_Published'].str[:10]
df2['Date_Published'] = df2['Date_Published'].astype('datetime64[ns]')

print(len(df2.index))
print(df2.groupby('Date_Published')['Title'].count())
#PLOT
df_count = df2.groupby('Date_Published')['Title'].count()
df_count.plot.line()
plt.show()
#NEW CSV
#df2.to_csv(export_data, encoding='utf-8', index=None)

####################################################################
'''
header = ['CNBC Press release', 'Reuters: Company News', 'Reuters: Business News', 'Top News and Analysis', 'WSJ_Financial_World']
df_count = pd.DataFrame(columns=header)
df_count['CNBC_Press_release'] = df2.groupby('Date_Published')['Section_title'].apply(lambda x: (x=='Press Releases - CNBC').sum())
df_count['Reuters: Company News'] = df2.groupby('Date_Published')['Section_title'].apply(lambda x: (x=='Reuters: Company News').sum())
df_count['Reuters: Business News'] = df2.groupby('Date_Published')['Section_title'].apply(lambda x: (x=='Reuters: Business News').sum())
df_count['Top News and Analysis'] = df2.groupby('Date_Published')['Section_title'].apply(lambda x: (x=='Top News and Analysis (pro)').sum())
df_count['WSJ_Financial_World'] = df2.groupby('Date_Published')['Section_title'].apply(lambda x: (x=='The Wall Street Journal &amp; Breaking News Business Financial and Economic News World News and Video').sum())

p1 = plt.bar(df_count['Date_Published'], df_count['CNBC_Press_release'], width, color='r')
p2 = plt.bar(df_count['Date_Published'], df_count['Reuters: Company News'], width, bottom=df_count['CNBC_Press_release'], color='b')
p3 = plt.bar(df_count['Date_Published'], df_count['Reuters: Business News'], width, bottom=df_count['Reuters: Company News'], color='g')
p4 = plt.bar(df_count['Date_Published'], df_count['Top News and Analysis'], width, bottom=df_count['Reuters: Business News'], color='c')

plt.ylim([0,120])
plt.yticks(fontsize=12)
plt.ylabel(output, fontsize=12)
plt.xticks(df_count['Date_Published'], X_AXIS, fontsize=12, rotation=90)
plt.xlabel('test', fontsize=12)
plt.legend((p1[0], p2[0], p3[0], p4[0]), (header[0], header[1], header[2], header[3]), fontsize=12, ncol=4, framealpha=0, fancybox=True)
plt.show()
'''