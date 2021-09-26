import json

filepath='data/conala/colana-train.json'
a=json.load(open(filepath))
f=open('data/conala/src.txt','w')

for i, data in a:
    f.write(data['intent']+'\n')
