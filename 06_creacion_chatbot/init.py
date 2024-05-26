from create_db import VectorDB
if __name__ == "__main__":
    vectordb = VectorDB()
    vectordb.loadDocuments('knowledge_database')
