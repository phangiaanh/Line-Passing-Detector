import os
from pymongo import MongoClient
import gridfs
import time


cluster = MongoClient("mongodb://PGA:111199@cluster0-shard-00-00.gjysk.mongodb.net:27017,cluster0-shard-00-01.gjysk.mongodb.net:27017,cluster0-shard-00-02.gjysk.mongodb.net:27017/<dbname>?ssl=true&replicaSet=atlas-8sqsyx-shard-0&authSource=admin&retryWrites=true&w=majority")
db = cluster["test"]
collection = db["test"]
fs = gridfs.GridFS(db)

while True:
    if(len(os.listdir('/home/ubuntu/mongodb images/')) == 0):
        time.sleep(0.1)
        continue

    listFiles = os.listdir('/home/ubuntu/mongodb images/')
    firstFile = listFiles[0]

    firstNameID = int(firstFile.split(".")[0])


    evidence = fs.put(open('/home/ubuntu/mongodb images/' + firstFile, 'rb'), filename = firstFile)

    collection.update_one({'_id': firstNameID}, {'$set': {'evidence': evidence}})

    print("Updated [_id]: ", evidence)

    os.remove('/home/ubuntu/mongodb images/' + firstFile)

    print(firstFile + ' deleted')
