def get_database():
    from dotenv import load_dotenv
    from pymongo import MongoClient
    import os
    import pymongo

    load_dotenv()

    CONNECTION_STRING = "mongodb+srv://" + os.getenv("USERNAME") + ":" + os.getenv("PASSWORD") + "@cv-project.btad0l1.mongodb.net/?retryWrites=true&w=majority"

    client = pymongo.MongoClient(CONNECTION_STRING)
    db = client.test
    
    return client['screen_time']
