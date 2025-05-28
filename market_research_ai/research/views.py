import os
import threading
import psycopg2
from django.shortcuts import render
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# === Configuration ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_CONFIG = {
    "dbname": "market_db",
    "user": "postgres",
    "password": "1329",
    "host": "localhost",
    "port": "5432"
}
os.environ["GOOGLE_API_KEY"] = "AIzaSyC0KzyEBjppn2n0aZxHxwhtRixi4NH8HEw"
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# === PG Connection ===
conn = psycopg2.connect(**DB_CONFIG)
register_vector(conn)

# === Lazy Embedder ===
_embedder = None
def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder

# === Retrieve Chunks from DB ===
def get_relevant_chunks(query, top_k=3):
    embedder = get_embedder()
    query_vec = embedder.encode([query])[0].tolist()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT content FROM your_table
            ORDER BY embedding <-> %s::vector
            LIMIT %s;
        """, (query_vec, top_k))
        results = cur.fetchall()
    return "\n\n".join(row[0] for row in results)

# === Gemini RAG Response ===
def generate_text_response(query):
    context = get_relevant_chunks(query)
    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = f"""
You are a market research expert. Use ONLY the provided context below to answer the question.

Context:
{context}

Question:
{query}

If the answer is not in the context, reply: "I don't have enough data to answer that."
"""
    response = model.generate_content(prompt)
    return response.text.strip()

# === Django View ===
def index(request):
    response = ""
    if request.method == "POST":
        query = request.POST.get("query", "").strip()

        def run_text():
            nonlocal response
            response = generate_text_response(query)

        t1 = threading.Thread(target=run_text)
        t1.start()
        t1.join()

    return render(request, "index.html", {
        "response": response,
    })


import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
def chat_api(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            query = data.get("message", "").strip()
            if not query:
                return JsonResponse({"error": "Empty message"}, status=400)

            response_text = generate_text_response(query)
            return JsonResponse({"reply": response_text})

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    return JsonResponse({"error": "POST method required."}, status=405)
