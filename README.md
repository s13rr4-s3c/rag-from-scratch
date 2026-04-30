# rag-from-scratch

Retrieval Augmented Generation implemented from scratch in Python — no LangChain, no LlamaIndex. PDF ingestion, semantic chunking, cosine similarity retrieval and OpenAI generation in a single readable pipeline.

---

## O que é RAG

Um LLM treinado não sabe o que está dentro do seu PDF.
Ele só sabe o que aprendeu durante o treinamento — e inventa o resto.

RAG resolve isso em três etapas:

```
1. INGESTION   → transforma o PDF em vetores semânticos e salva
2. RETRIEVAL   → quando você pergunta algo, busca os trechos mais relevantes
3. GENERATION  → injeta esses trechos no contexto do LLM antes de gerar a resposta
```

O LLM deixa de "inventar" e passa a responder com base no documento real.

---

## Arquitetura

```
┌─────────────────────────────────────────────────────────────┐
│                        INGESTION                            │
│  PDF ──► extract_text ──► split_chunks ──► embeddings       │
│                                                ↓            │
│                                  vector_store.sqlite (SQLite+BLOB) │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      RETRIEVAL                              │
│  pergunta ──► embedding ──► cosine_similarity ──► top_k     │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      GENERATION                             │
│  system_prompt + chunks + pergunta ──► LLM ──► resposta     │
└─────────────────────────────────────────────────────────────┘
```

Documentação detalhada: [docs/architecture.md](docs/architecture.md)

---

## Estrutura

```
rag-from-scratch/
├── main.py              # CLI — ponto de entrada
├── src/
│   └── rag.py           # pipeline completo comentado
├── tests/
│   └── test_rag.py      # testes unitários (sem precisar de chave OpenAI)
├── docs/
│   └── architecture.md  # decisões de arquitetura explicadas
├── data/                # vector store gerado localmente (no .gitignore)
├── requirements.txt
└── .gitignore
```

---

## Setup

**1. Clone e instale dependências**

```bash
git clone https://github.com/s13rr4-s3c/rag-from-scratch.git
cd rag-from-scratch
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**2. Configure a chave da OpenAI**

```bash
export OPENAI_API_KEY="sk-..."
```

**3. Execute com qualquer PDF**

```bash
python main.py --pdf caminho/para/documento.pdf
```

---

## Uso

```bash
# Modo básico
python main.py --pdf documento.pdf

# Ver quais trechos foram recuperados
python main.py --pdf documento.pdf --verbose

# Forçar reindexação quando trocar o PDF
python main.py --pdf novo_documento.pdf --rebuild
```

---

## Testes

```bash
pytest tests/test_rag.py -v
```

---

## Conceitos implementados

| Conceito | Onde | O que faz |
|----------|------|-----------|
| Text chunking com overlap | `split_into_chunks()` | Divide o PDF preservando contexto nas bordas |
| Embeddings semânticos | `generate_embeddings()` | Transforma texto em vetores de 1536 dimensões |
| Similaridade de cosseno | `cosine_similarity()` | Mede relevância semântica independente da magnitude |
| Vector store persistente (SQLite+BLOB) | `build_vector_store()` | Persiste embeddings e chunks em SQLite para reutilização eficiente |
| Prompt engineering | `generate_answer()` | Instrui o LLM a usar apenas o contexto fornecido |

---

## Limitações intencionais

Esta implementação é didática. Em produção você precisaria de:

- **Vector database** (Chroma, pgvector, Pinecone) no lugar do SQLite local para escalar a milhões de chunks
- **Índice ANN** (HNSW) para escalar a milhões de chunks
- **Chunking semântico** respeitando parágrafos e seções
- **Reranker** para refinar os resultados do retrieval
- **Ingestão incremental** para atualizar sem reprocessar tudo

---

## Custo estimado

| Operação | Modelo | Custo aproximado |
|----------|--------|-----------------|
| Indexar PDF de 50 páginas | text-embedding-3-small | ~$0.001 |
| Cada pergunta (retrieval) | text-embedding-3-small | ~$0.000004 |
| Cada resposta (geração) | gpt-4o-mini | ~$0.001 |

---

## Referências

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — paper original do Transformer
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [RAG (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401) — paper original do RAG

---