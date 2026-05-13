import os
from dotenv import load_dotenv
load_dotenv()

print("Testing Gemini...")
from google import genai
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Reply with just: Gemini working."
)
print(response.text.strip())

print("Testing ChromaDB...")
import chromadb
col = chromadb.Client().create_collection("test2")
col.add(documents=["hello"], ids=["1"])
print("ChromaDB:", col.query(query_texts=["hello"], n_results=1)["documents"][0][0])

print("\nAll systems go. Ready for Day 1.")
