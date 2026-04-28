# Arquitetura do RAG

## Visão geral

```
┌─────────────────────────────────────────────────────────────┐
│                        INGESTION                            │
│                   (roda uma vez por PDF)                    │
│                                                             │
│  PDF ──► extract_text ──► split_chunks ──► embeddings       │
│                                                ↓            │
│                                        vector_store.pkl     │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      RETRIEVAL                              │
│                  (roda a cada pergunta)                     │
│                                                             │
│  pergunta ──► embedding ──► cosine_similarity ──► top_k     │
│                                                    chunks   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      GENERATION                             │
│                                                             │
│  system_prompt + chunks + pergunta ──► LLM ──► resposta     │
└─────────────────────────────────────────────────────────────┘
```

## Por que cada decisão foi tomada assim

### Chunk size 500 chars com overlap 50

Chunks muito pequenos perdem contexto. Chunks muito grandes desperdiçam tokens do contexto do LLM.
500 chars (~100 palavras) é um parágrafo médio — unidade natural de informação.
O overlap de 50 chars garante que informações na borda entre dois chunks não sejam perdidas.

### Por que similaridade de cosseno

O embedding tem 1536 dimensões. A distância euclidiana seria distorcida pela magnitude dos vetores — textos mais longos geram vetores numericamente maiores. O cosseno mede apenas a direção, capturando o significado independente do tamanho.

### Por que pickle para o vector store

Para uma PoC, pickle é suficiente e sem dependências extras. Em produção, usaria Chroma, Pinecone, Weaviate ou pgvector — databases vetoriais que escalam para milhões de chunks e suportam índices ANN para buscas mais rápidas.

### Por que temperature 0.1 na geração

Temperature controla a aleatoriedade na geração. Para RAG factual queremos respostas reproduzíveis e fiéis ao documento, então temperature baixa é mais adequada.

## O que seria diferente em produção

| PoC | Produção |
|-----|----------|
| pickle para vector store | Chroma / pgvector / Pinecone |
| busca linear O(n) | índice HNSW O(log n) |
| chunks por tamanho fixo | chunking semântico (por parágrafo/seção) |
| sem cache | cache de embeddings de queries frequentes |
| single PDF | ingestão incremental de múltiplos docs |
| sem reranking | cross-encoder reranker após retrieval |
| prompt fixo | prompt dinâmico por domínio |
```
