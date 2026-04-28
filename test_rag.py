"""
Testes unitários para os componentes do RAG.

Não precisam de chave OpenAI — testam a lógica pura.

Executar:
  pytest tests/test_rag.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag import cosine_similarity, split_into_chunks


# ─── split_into_chunks ────────────────────────────────────────────────────────

class TestSplitIntoChunks:
    def test_texto_menor_que_chunk_vira_um_unico_chunk(self):
        text = "texto curto"
        chunks = split_into_chunks(text, chunk_size=100, overlap=10)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_texto_exato_do_chunk_size(self):
        text = "a" * 500
        chunks = split_into_chunks(text, chunk_size=500, overlap=0)
        assert len(chunks) == 1

    def test_overlap_gera_mais_chunks_que_sem_overlap(self):
        text = "x" * 1000
        sem_overlap = split_into_chunks(text, chunk_size=100, overlap=0)
        com_overlap = split_into_chunks(text, chunk_size=100, overlap=20)
        assert len(com_overlap) > len(sem_overlap)

    def test_chunks_vazios_sao_ignorados(self):
        text = "   \n\n   \n   "
        chunks = split_into_chunks(text, chunk_size=50, overlap=0)
        assert len(chunks) == 0

    def test_chunks_tem_tamanho_maximo_correto(self):
        text = "palavra " * 200
        chunk_size = 100
        chunks = split_into_chunks(text, chunk_size=chunk_size, overlap=0)
        for chunk in chunks:
            assert len(chunk) <= chunk_size


# ─── cosine_similarity ────────────────────────────────────────────────────────

class TestCosineSimilarity:
    def test_vetores_identicos_tem_similaridade_1(self):
        v = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-6)

    def test_vetores_opostos_tem_similaridade_negativa(self):
        v1 = np.array([1.0, 0.0])
        v2 = np.array([-1.0, 0.0])
        assert cosine_similarity(v1, v2) == pytest.approx(-1.0, abs=1e-6)

    def test_vetores_perpendiculares_tem_similaridade_0(self):
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        assert cosine_similarity(v1, v2) == pytest.approx(0.0, abs=1e-6)

    def test_similaridade_independe_da_magnitude(self):
        """Vetores na mesma direção mas magnitudes diferentes → similaridade 1."""
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = v1 * 100
        assert cosine_similarity(v1, v2) == pytest.approx(1.0, abs=1e-6)

    def test_resultado_esta_no_intervalo_valido(self):
        rng = np.random.default_rng(42)
        for _ in range(50):
            v1 = rng.standard_normal(128)
            v2 = rng.standard_normal(128)
            sim = cosine_similarity(v1, v2)
            assert -1.0 - 1e-6 <= sim <= 1.0 + 1e-6
