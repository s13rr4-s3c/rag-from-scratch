"""
RAG (Retrieval Augmented Generation) - Implementação Core
=========================================================
Fluxo:
  1. Ingestion  → carrega PDF, divide em chunks, gera embeddings, salva no vector store
  2. Retrieval  → converte pergunta em embedding, busca chunks mais similares
  3. Generation → injeta chunks no contexto do LLM e gera resposta
"""

import os
import pickle
from pathlib import Path

import numpy as np
from openai import OpenAI
from pypdf import PdfReader

# ─── Configuração ────────────────────────────────────────────────────────────

CHUNK_SIZE = 500          # tamanho de cada chunk em caracteres
CHUNK_OVERLAP = 50        # sobreposição entre chunks (evita perda de contexto na borda)
TOP_K = 4                 # quantos chunks recuperar por pergunta
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
VECTOR_STORE_PATH = Path("data/vector_store.pkl")

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


# ─── Etapa 1: Ingestion ───────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extrai texto bruto de um PDF página por página."""
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            pages.append(f"[Página {i+1}]\n{text}")
    return "\n\n".join(pages)


def split_into_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Divide o texto em chunks com sobreposição.

    Por que overlap?
    Se uma informação importante cai exatamente na borda entre dois chunks,
    sem overlap ela ficaria cortada pela metade em ambos.
    Com overlap, ela aparece completa em pelo menos um dos chunks.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def generate_embeddings(chunks: list[str]) -> list[list[float]]:
    """
    Converte cada chunk em um vetor numérico (embedding).

    O embedding captura o *significado semântico* do texto.
    Textos com significado parecido ficam próximos no espaço vetorial,
    mesmo usando palavras diferentes.

    Ex: "como funciona transistor" e "princípio do semicondutor"
    ficam próximos — sem nenhuma palavra em comum.
    """
    print(f"  Gerando embeddings para {len(chunks)} chunks...")
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=chunks
    )
    return [item.embedding for item in response.data]


def build_vector_store(pdf_path: str, force_rebuild: bool = False) -> dict:
    """
    Pipeline completo de ingestion.
    Salva o vector store em disco para não reprocessar o PDF toda vez.
    """
    VECTOR_STORE_PATH.parent.mkdir(exist_ok=True)

    if VECTOR_STORE_PATH.exists() and not force_rebuild:
        print("✓ Vector store já existe. Carregando do disco...")
        with open(VECTOR_STORE_PATH, "rb") as f:
            return pickle.load(f)

    print(f"→ Processando PDF: {pdf_path}")

    print("  [1/3] Extraindo texto...")
    text = extract_text_from_pdf(pdf_path)
    print(f"  {len(text):,} caracteres extraídos")

    print("  [2/3] Dividindo em chunks...")
    chunks = split_into_chunks(text)
    print(f"  {len(chunks)} chunks criados")

    print("  [3/3] Gerando embeddings...")
    embeddings = generate_embeddings(chunks)

    vector_store = {
        "chunks": chunks,
        "embeddings": np.array(embeddings),  # matriz (n_chunks × embedding_dim)
        "source": pdf_path,
    }

    with open(VECTOR_STORE_PATH, "wb") as f:
        pickle.dump(vector_store, f)

    print(f"✓ Vector store salvo em {VECTOR_STORE_PATH}")
    return vector_store


# ─── Etapa 2: Retrieval ───────────────────────────────────────────────────────

def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Similaridade de cosseno entre dois vetores.

    Mede o ângulo entre eles — não a distância absoluta.
    Valor 1.0  = vetores idênticos em direção (mesmo significado)
    Valor 0.0  = vetores perpendiculares (sem relação)
    Valor -1.0 = vetores opostos

    Por que cosseno e não distância euclidiana?
    Textos longos geram vetores maiores, mas o *significado* é
    capturado pela direção, não pelo tamanho do vetor.
    """
    return float(
        np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    )


def retrieve(query: str, vector_store: dict, top_k: int = TOP_K) -> list[dict]:
    """
    Recupera os chunks mais relevantes para a pergunta.

    1. Converte a pergunta em embedding (mesmo espaço vetorial dos chunks)
    2. Calcula similaridade com todos os chunks
    3. Retorna os top_k mais similares
    """
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[query]
    )
    query_embedding = np.array(response.data[0].embedding)

    similarities = [
        cosine_similarity(query_embedding, chunk_emb)
        for chunk_emb in vector_store["embeddings"]
    ]

    top_indices = np.argsort(similarities)[::-1][:top_k]

    return [
        {
            "chunk": vector_store["chunks"][i],
            "score": similarities[i],
            "index": int(i),
        }
        for i in top_indices
    ]


# ─── Etapa 3: Generation ─────────────────────────────────────────────────────

def generate_answer(query: str, retrieved_chunks: list[dict]) -> str:
    """
    Gera a resposta final injetando os chunks recuperados no contexto do LLM.

    Isso é o RAG em ação:
    - Sem RAG: LLM responde com o que aprendeu no treinamento (pode alucinar)
    - Com RAG: LLM responde baseado nos trechos reais do documento
    """
    context = "\n\n---\n\n".join(
        f"[Trecho {i+1} | relevância: {r['score']:.3f}]\n{r['chunk']}"
        for i, r in enumerate(retrieved_chunks)
    )

    system_prompt = """Você é um assistente especializado em responder perguntas
com base exclusivamente nos trechos de documento fornecidos.

Regras:
- Use APENAS as informações dos trechos fornecidos
- Se a informação não estiver nos trechos, diga explicitamente
- Cite de qual trecho veio cada informação quando possível
- Seja preciso e direto"""

    user_prompt = f"""Trechos recuperados do documento:
{context}

Pergunta: {query}"""

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )

    return response.choices[0].message.content


# ─── Interface principal ──────────────────────────────────────────────────────

def ask(query: str, vector_store: dict, verbose: bool = False) -> str:
    """
    Pipeline completo: pergunta → retrieval → geração → resposta.
    """
    retrieved = retrieve(query, vector_store)

    if verbose:
        print(f"\n{'─'*50}")
        print(f"Chunks recuperados (top {TOP_K}):")
        for r in retrieved:
            print(f"  [score: {r['score']:.3f}] {r['chunk'][:80]}...")
        print(f"{'─'*50}\n")

    return generate_answer(query, retrieved)
