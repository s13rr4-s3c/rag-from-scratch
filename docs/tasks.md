Parte 1 — Core PDF RAG (implementado)
[x] Implementar CLI básica (main.py) para indexação e consulta de PDF.
[x] Implementar extração de texto de PDF via pypdf.
[x] Implementar chunking do texto com overlap.
[x] Gerar embeddings com OpenAI API e persistir localmente (pickle).
[x] Implementar vetor store simples em arquivo (data/vector_store.pkl).
[x] Implementar retrieval com similaridade de cosseno.
[x] Gerar respostas com OpenAI Chat, usando system prompt restritivo.
[x] Adicionar opção de reindexação forçada via flag --rebuild.
[x] Adicionar modo verbose que mostra chunks recuperados.
[x] Ignorar data/ no versionamento.
[x] Cobrir pipeline básico com testes unitários usando pytest.
[x] Documentar arquitetura, limitações, comandos e dependências.
Parte 2 — Evolução e Generalização (planejado)
[x] Migrar persistência do vetor store de pickle para SQLite + numpy. (Concluído: gravação, leitura, busca e resposta totalmente via SQLite, testado em ciclo CLI e testes unitários)
[ ] Adicionar suporte a metadados por chunk (ex: source, page, type, name).
[ ] Permitir ingestão incremental (não reprocessar arquivos já indexados).
[ ] Modularizar código: separar components (vector_store.py, ingestors.py).
[ ] Implementar ingestors para múltiplos formatos: PDF, texto (.md, .txt, .rst), código (.py, .yaml, .json, .tf etc).
[ ] Implementar chunking semântico para *.py via ast (função/classe).
[ ] Permitir ingestão de diretórios recursivamente, com filtro por extensão.
[ ] Implementar deduplicação de chunks (hash de chunk).
[ ] Adicionar flags CLI: --dir, --ext, etc.
[ ] Atualizar documentação e exemplos de uso.
Parte 3 — Gap Analysis & Auditoria (planejado)
[ ] Implementar gap analysis: dado um requisito, buscar evidências documentais.
[ ] Definir e aplicar threshold de similaridade para limitar falsos positivos.
[ ] Exportar relatórios de auditoria (Markdown e CSV).
[ ] Implementar system prompt ainda mais restritivo para modo auditoria.
[ ] Garantir auditabilidade total (resposta rastreável até chunk e fonte).
[ ] Permitir filtros de busca em fontes/arquivos específicos.
[ ] Adicionar flags CLI: --mode=query, --mode=audit, --requirements, --report, --filter, etc.
[ ] Documentar novos fluxos, limitações e critérios de pronto.
[ ] Cobrir novas funções com testes unitários.
Padrões de Código & Segurança (contínuo)
[ ] Garantir type hints e docstrings em todas as funções públicas.
[ ] Manter funções com responsabilidade única e até ~30 linhas.
[ ] Manter tratamento explícito de erros.
[ ] Manter dados sensíveis fora do repositório (data/ no .gitignore).
[ ] Verificar testes unitários para toda funcionalidade exposta.
