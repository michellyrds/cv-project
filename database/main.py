def get_database(database: str):
    from dotenv import load_dotenv
    from pymongo import MongoClient
    import os

    load_dotenv()

    CONNECTION_STRING = "mongodb+srv://" + os.getenv("MONGOUSERNAME") + ":" + os.getenv("PASSWORD") + "@cv-project.btad0l1.mongodb.net/?retryWrites=true&w=majority"
    print(CONNECTION_STRING)
    client = MongoClient(CONNECTION_STRING)
    db = client.get_database(database)
    print(db)
    return db
