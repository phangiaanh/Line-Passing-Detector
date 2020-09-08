import pymongo
from pymongo import MongoClient

cluster = MongoClient("mongodb://PGA:111199@cluster0-shard-00-00.gjysk.mongodb.net:27017,cluster0-shard-00-01.gjysk.mongodb.net:27017,cluster0-shard-00-02.gjysk.mongodb.net:27017/<dbname>?ssl=true&replicaSet=atlas-8sqsyx-shard-0&authSource=admin&retryWrites=true&w=majority")
db = cluster["test"]
collection = db["test"]

collection.delete_many({})

collection = db["fs.chunks"]
collection.delete_many({})
collection = db["fs.files"]
collection.delete_many({})
