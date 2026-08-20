"""
hash_strategy.py — Estratégias de hash plugáveis para o cache.

O cache NUNCA usa hash() nativo do Python, que não é estável entre processos.
Todas as estratégias devem ser determináveis, portáveis e thread-safe.
"""
import hashlib
from abc import ABC, abstractmethod


class HashStrategy(ABC):
    """Interface para estratégia de hash."""

    @abstractmethod
    def hash(self, data: bytes) -> int:
        """
        Calcula hash dos bytes e retorna um int não-negativo.

        Args:
            data: bytes da assinatura espectral.

        Returns:
            int não-negativo, sem limite de tamanho definido pela interface.
            O cache fará `% n_slots` sobre este valor.
        """
        ...


class Blake2bHashStrategy(HashStrategy):
    """
    Hash usando blake2b com digest de 8 bytes.
    Não depende de dependência externa. Default quando xxhash não disponível.
    """

    def hash(self, data: bytes) -> int:
        digest = hashlib.blake2b(data, digest_size=8).digest()
        return int.from_bytes(digest, byteorder="little", signed=False)


class XxHashStrategy(HashStrategy):
    """
    Hash usando xxhash (xxh64). Mais rápido que blake2b.
    Requer pacote xxhash instalado.
    """

    def hash(self, data: bytes) -> int:
        import xxhash  # lazy import: erro claro se não instalado
        return xxhash.xxh64_intdigest(data)


class ConstantHashStrategy(HashStrategy):
    """
    Sempre retorna 0. Força todos os slots para o índice 0 % n_slots.
    SOMENTE para testes de colisão forçada. Nunca usar em produção.
    """

    def hash(self, data: bytes) -> int:
        return 0
