import numpy as np
import requests
from sentence_transformers import SentenceTransformer
import faiss

# -----------------------------
# 1) Base recuperável
# -----------------------------
DOCUMENTOS = [
    "Fruto da mangueira, com um grande caroço central que envolve sua semente, "
    "muito conhecido por sua polpa amarelada, doce e suculenta."
]

# -----------------------------
# 2) Modelo de embeddings 
# -----------------------------
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Gera embeddings dos documentos
doc_emb = embedder.encode(DOCUMENTOS, normalize_embeddings=True)
doc_emb = np.asarray(doc_emb, dtype=np.float32)

# -----------------------------
# 3) Indexação vetorial (FAISS)
# -----------------------------
dim = doc_emb.shape[1]
index = faiss.IndexFlatIP(dim)  # IP + vetores normalizados ~= similaridade cosseno
index.add(doc_emb)

# -----------------------------
# 4) Consulta + recuperação top-k
# -----------------------------
pergunta = "O que é Manga?"
q_emb = embedder.encode([pergunta], normalize_embeddings=True)
q_emb = np.asarray(q_emb, dtype=np.float32)

top_k = 2
scores, ids = index.search(q_emb, top_k)

recuperados = [DOCUMENTOS[i] for i in ids[0].tolist() if i != -1]
contexto = "\n\n".join(
    f"[Documento {n+1}]\n{txt}" for n, txt in enumerate(recuperados)
)

# -----------------------------
# 5) Geração (Ollama llama3:8b) com contexto recuperado
# -----------------------------
prompt = (
    "Responda usando APENAS o contexto.\n"
    "Se existir mais de um sentido possível para o termo, escolha o sentido que melhor "
    "responde à pergunta e diga qual sentido foi adotado.\n\n"
    f"Contexto:\n{contexto}\n\n"
    f"Pergunta: {pergunta}\n"
    "Resposta:"
)

payload = {
    "model": "llama3:8b",
    "prompt": prompt,
    "stream": False
}

resp = requests.post("http://localhost:11434/api/generate", json=payload, timeout=120)
resp.raise_for_status()
print(resp.json().get("response", "").strip())
