def get_database():
    from pymongo import MongoClient
    import pymongo

    CONNECTION_STRING = "mongodb+srv://<username>:<password>@cv-project.btad0l1.mongodb.net/?retryWrites=true&w=majority"

    client = pymongo.MongoClient(CONNECTION_STRING)
    db = client.test
    print(db)
    
    return client['screen_time']

if __name__ == "__main__":

    dbname = get_database()
    print(dbname)
