import os
import json

risksofbias = {}

for file in os.listdir('robotreviewer_report'):
    with open('robotreviewer_report/'+file,'r', encoding="utf8") as infile:
        jsondata = json.load(infile)
        for i in range(len(jsondata['article_data'])):
            filename = jsondata['article_data'][i]['gold']['filename']
            riskofb = [ (x['domain'],x['judgement']) for x in jsondata['article_data'][1]['ml']['bias'] ]
            risksofbias[filename] = riskofb