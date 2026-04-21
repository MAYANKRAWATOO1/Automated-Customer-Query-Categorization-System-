import pandas as pd
from model import QueryClassifier
from faiss_db import FAISSDatabase
from chatbot import ChatbotEngine

def main():

    df = pd.read_csv("dataset.csv")

    print("Training models...")
    classifier = QueryClassifier()
    classifier.train(df)

    print("Building FAISS...")
    vectors = classifier.get_all_vectors(df['Query'].tolist())
    faiss_db = FAISSDatabase()
    faiss_db.build(df, vectors)

    chatbot = ChatbotEngine()

    print("\n✅ SYSTEM READY\n")

    while True:
        query = input("You: ")

        if query.lower() in ["exit", "quit"]:
            break

        # ML
        ml = classifier.predict(query)

        print("\nML Results:")
        print("LR:", ml["logistic_regression"])
        print("SVM:", ml["svm"])
        print("ANN:", ml["ann"])
        print("Cluster:", ml["kmeans_cluster"])

        # FAISS
        results = faiss_db.search(ml["feature_vector"], k=1)

        if results:
            context = results[0]["response"]
        else:
            context = "Please contact support."

        # Chatbot
        response, mode = chatbot.generate_response(
            query,
            ml["final_category"],
            context
        )

        print("\nBot:", response)
        print("-" * 40)

if __name__ == "__main__":
    main()