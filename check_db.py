import os
from pymongo import MongoClient
from dotenv import load_dotenv
from mongodb_tunnel import start_ssh_tunnel

load_dotenv()
start_ssh_tunnel()
client = MongoClient(os.getenv('MONGODB_URI'))
db = client[os.getenv('MONGODB_DATABASE', 'sahamify_db')]
print(f"Collections: {db.list_collection_names()}")
coll = db['fundamental_data']
print(f"Count: {coll.count_documents({})}")
for doc in coll.find().limit(5):
    print(doc.get('ticker'))
client.close()
