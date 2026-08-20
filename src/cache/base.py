"""
base.py — Interface abstrata do cache.

Qualquer implementação de cache (NullCache, SharedDirectMappedCache, ou futura
set-associative) deve satisfazer este contrato.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass


import numpy as np


@dataclass
class LookupStats:
    """Estatísticas de uma operação lookup_many."""
    n_queries: int = 0
    n_hits: int = 0
    n_misses_empty: int = 0
    n_misses_collision: int = 0


@dataclass
class InsertStats:
    """Estatísticas de uma operação insert_many."""
    n_attempts: int = 0
    n_inserted: int = 0
    n_skipped_collision: int = 0


class BaseCache(ABC):
    """
    Interface do cache de assinaturas espectrais.

    Contrato:
        lookup_many e insert_many são as únicas APIs que a pipeline usa.
        A pipeline nunca acessa slots diretamente.
    """

    @abstractmethod
    def lookup_many(
        self,
        key_array: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, LookupStats]:
        """
        Consulta o cache para múltiplas assinaturas.

        Args:
            key_array: array (N, n_bands), dtype preservado (uint8/uint16/float32),
                       C-contiguous. Gerado por make_cache_key_values.

        Returns:
            values: array (N,) dtype uint8. Válido apenas onde hit_mask == True.
            hit_mask: array (N,) dtype bool. True onde houve hit.
            stats: LookupStats com contagens desta chamada.

        Garantia:
            values[~hit_mask] pode conter qualquer valor (não inicializado).
            A pipeline NUNCA usa values onde hit_mask == False.
        """
        ...

    @abstractmethod
    def insert_many(
        self,
        key_array: np.ndarray,
        values: np.ndarray,
    ) -> InsertStats:
        """
        Insere pares (assinatura, valor) no cache.

        Args:
            key_array: array (N, n_bands), mesmo formato de lookup_many.
            values: array (N,) dtype uint8 com os valores a inserir.

        Returns:
            InsertStats com contagens desta chamada.

        Política (write_on_empty):
            Slot vazio → grava.
            Slot com mesma assinatura → considera já inserido (sem reescrita).
            Slot com assinatura diferente → COLISÃO, não sobrescreve, incrementa
            n_skipped_collision.
        """
        ...

    @property
    @abstractmethod
    def enabled(self) -> bool:
        """True se o cache está ativo. NullCache retorna False."""
        ...
