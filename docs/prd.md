# rag-from-scratch — Documento de Requisitos do Produto (PRD)

## 1. Visão Geral

Projeto de Retrieval Augmented Generation (RAG) construído do zero em Python, sem frameworks externos. Permite responder perguntas sobre documentos PDF fornecidos pelo usuário, sempre baseando as respostas exclusivamente no conteúdo desses arquivos.

**Casos de uso principais:**
- Consulta livre/interativa sobre PDFs
- (Planejado) Gap analysis/auditoria de requisitos

**Princípios:**
- Resposta estritamente fundamentada nos documentos: o modelo nunca deve usar conhecimento externo.

**Deliberações/limitações:**
- Não utiliza LangChain, LlamaIndex, bancos vetoriais externos ou dados externos.
- Nenhuma resposta embute conhecimento que não esteja explícito nos documentos indexados.

---

## 2. Metas do Projeto

- **Funcionais:** Consulta a PDFs (presente), futura ingestão de múltiplos formatos, gap analysis/auditoria (planejado).
- **Qualidade:** Código limpo, testável (pytest), sem dependências desnecessárias.
- **Confiabilidade:** Deve evitar alucinações do LLM; se não houver evidência, informar ausência de resposta.
- **Evolução:** Estrutura modular para servir como template para outros projetos de RAG.

---

## 3. Stack Tecnológica

- **Linguagem:** Python ≥ <!-- TODO: verificar versão mínima -->
- **Dependências atuais:**
  - openai ≥1.30.0 (embeddings + LLM)
  - pypdf ≥4.0.0 (extração de PDF)
  - numpy ≥1.26.0 (operações vetoriais)
  - pytest ≥8.0.0 (testes)
- **Exclusões:** Nada de ORMs, LangChain, LlamaIndex, bancos vetoriais, etc.
- **Ferramentas de teste:** pytest
- **Recursos planejados:** SQLite, ingestors múltiplos, etc. (ainda não implementados).

---

## 4. Modelos de Dados

**Vector Store (atual):**
```python
{
  "chunks": [str],
  "embeddings": np.ndarray,
  "source": str  # caminho do PDF
}
```
- Serializado em data/vector_store.pkl via pickle.
- Não há metadados detalhados por chunk (p. ex.: página, tipo, nome) <!-- TODO: verificar quando parte 2 for implementada -->.
- Embeddings: matriz numpy de (n_chunks × embedding_dim).

**Fluxo de dados:**
PDF → extract_text_from_pdf → split_into_chunks → generate_embeddings → dict → pickle/file

**Estruturas intermediárias:**
- retrieved_chunks: lista de dicts {"chunk": texto, "score": float, "index": int}

---

## 5. Segurança

- **Restrições no contexto:** System prompt instrui o modelo a usar apenas os trechos recuperados. Temperature 0.1 para evitar variações criativas.
- **Threshold de similaridade:** Não existe threshold configurável por enquanto <!-- TODO: implementar e documentar -->.
- **Deduplicação:** Não existe ainda <!-- TODO: implementar -->.
- **Sanitização de inputs:** Não há segurança ativa para path traversal, arquivos binários etc. <!-- TODO: implementar -->.
- **Chave de API:** OPENAI_API_KEY obrigatória; nunca hardcoded.
- **Dados sensíveis:** data/ está no .gitignore — previne vazamento de persistências.
- **Auditabilidade:** As respostas podem ser rastreadas até os chunks, mas não até páginas ou arquivos múltiplos <!-- TODO: expandir com metadados e rastreabilidade futura -->.

---

## 6. Arquitetura de Módulos

- **main.py:** CLI/orquestração do pipeline
- **rag.py:** pipeline principal (ingestão, retrieval, geração, utilitários)
- Não há módulos separados para vector store, ingestors, gap_analysis (planejado).
- **Dependências:** main.py importa funções públicas build_vector_store e ask de rag.py.
- **Público vs privado:** build_vector_store/ask públicas.
- **Orquestração:** main.py manipula ingestão, loop de perguntas, flags CLI.

---

## 7. Interface CLI

- **Existente:**
  - --pdf (obrigatório): caminho para o PDF
  - --rebuild: força reindexação
  - --verbose: mostra chunks recuperados
- **Planejado (não existe ainda):**
  - --dir, --ext, --mode, --requirements, --report, --filter, etc. <!-- TODO: implementar e documentar posteriormente -->

---

## 8. Plano de Implementação

**Parte 1 (presente):**
- Pré-requisito: OPENAI_API_KEY definida.
- Arquivos criados: data/vector_store.pkl
- Arquivos modificados: main.py, rag.py
- Testes: test_rag.py
- Pronto quando: consulta interativa a PDF funcional, com test coverage.

**Parte 2 (planejada):**
- Migrar persistência para SQLite, adicionar metadados, múltiplos ingestors, ingestão incremental.
- Arquivos e funções a modularizar/distintas <!-- TODO: detalhar quando iniciar parte 2 -->.

**Parte 3 (planejada):**
- Gap analysis, thresholds de cobertura, exportação de relatório, rastreabilidade e auditoria.
- <!-- TODO: detalhar -->

---

## 9. Padrões de Código

- Todas funções públicas devem ter type hints e docstrings <!-- TODO: verificar cobertura -->
- Funções curtas, responsabilidade única.
- Efeitos colaterais explícitos em nome/assinatura.
- Tratamento de erro explícito — sem except solto ou “pass”.
- Constantes sempre no topo dos arquivos.
- Testes unitários para cada nova função pública.
