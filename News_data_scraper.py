#This code searches in the parent directory for all five unzipped folders, January to May and collects the information 
#needed from the json meta data
#I removed potential commas from the headline texts and the section title
#Data used: https://www.kaggle.com/jeet2016/us-financial-news-articles
import pandas as pd
import os
import json


path = "D:/Python/Data/US_Financial_News"
export_data = "D:/Python/Data/US_Financial_News/2018_01_05.csv"
os.chdir(path)
 
tit = []
pub = []
sectit = []
sit = []

for root, dirs, files in os.walk(path):
	for name in files:
		filename = os.path.join(root, name)
		with open(filename, encoding='utf-8') as fh:
			data = json.load(fh)
		title = data["title"]
		section_title = data["thread"]["section_title"]
		tit.append(title.replace(',',''))
		pub.append(data["published"])
		sectit.append(section_title.replace(',', ''))
		sit.append(data["thread"]["site"])
	print("Batch {} successfully processed".format( root))

#for filename in os.listdir(path):
#	with open(filename, encoding='utf-8') as fh:
#		data = json.load(fh)
#	tit.append(data["title"])
#	pub.append(data["published"])
#	sectit.append(data["thread"]["section_title"])
#	sit.append(data["thread"]["site"])

#print(data)
#print(data["section_title"])

df = pd.DataFrame(list(zip(tit, pub, sectit, sit)), columns =['Title', 'Date_Published', 'Section_title', 'Site'])

df.to_csv(export_data, encoding='utf-8', index=None)
