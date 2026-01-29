import os
from pymongo import MongoClient
from dotenv import load_dotenv
from mongodb_tunnel import start_ssh_tunnel

load_dotenv()
start_ssh_tunnel()
client = MongoClient(os.getenv('MONGODB_URI'))
db = client[os.getenv('MONGODB_DATABASE', 'sahamify_db')]
coll = db['fundamental_data']
doc = coll.find_one()
if doc:
    print(f"Top level keys: {list(doc.keys())}")
    if 'ticker' in doc:
        print(f"Ticker value: {doc['ticker']}")
    if 'kode' in doc:
        print(f"Kode value: {doc['kode']}")
client.close()
