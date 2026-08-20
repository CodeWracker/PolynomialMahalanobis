"""
shared_direct_mapped.py — Cache direct-mapped em shared memory.

ATENÇÃO: Esta classe usa multiprocessing.shared_memory.SharedMemory.
Workers que precisam usar o cache devem receber o nome da shared memory
(shm.name) e o número de slots via init_worker, e anexar a memória lá.
Nunca passar a instância de SharedDirectMappedCache via pickle, pois isso
NÃO compartilha a memória — apenas serializa os metadados.
"""
import math
import multiprocessing
import multiprocessing.synchronize  # noqa: F401 — tipo usado em hint: multiprocessing.synchronize.Lock
import numpy as np
from multiprocessing import shared_memory


from .base import BaseCache, LookupStats, InsertStats
from .hash_strategy import HashStrategy, Blake2bHashStrategy

_DEFAULT_N_LOCKS: int = 256


class SharedDirectMappedCache(BaseCache):
    """
    Cache direct-mapped em shared memory.

    Uso em workers (spawn):
        1. No processo principal, criar: cache = SharedDirectMappedCache(...)
        2. Passar cache.shm_name, cache.n_slots, cache.n_bands, cache.sig_size,
           cache.slot_size, cache.locks para init_worker via initargs.
        3. Em init_worker, chamar: SharedDirectMappedCache.attach(...)
    """

    def __init__(
        self,
        size_mb: float,
        n_bands: int,
        sig_dtype: np.dtype,
        hash_strategy: HashStrategy | None = None,
        n_locks: int = _DEFAULT_N_LOCKS,
    ) -> None:
        if hash_strategy is None:
            hash_strategy = Blake2bHashStrategy()

        self._hash = hash_strategy
        self.n_bands = n_bands
        self.sig_dtype = np.dtype(sig_dtype)
        self.sig_size = n_bands * self.sig_dtype.itemsize
        self.slot_size = 1 + self.sig_size + 1

        requested_bytes = int(size_mb * 1024 * 1024)
        self.n_slots = max(1, math.floor(requested_bytes / self.slot_size))
        self.effective_bytes = self.n_slots * self.slot_size

        self._shm = shared_memory.SharedMemory(
            create=True, size=max(self.effective_bytes, 1)
        )
        self.shm_name = self._shm.name

        buf = np.ndarray(self.effective_bytes, dtype=np.uint8, buffer=self._shm.buf)
        buf[:] = 0

        self.n_locks = n_locks
        ctx = multiprocessing.get_context("spawn")
        self.locks = [ctx.Lock() for _ in range(n_locks)]

        self._buf = np.ndarray(
            self.effective_bytes, dtype=np.uint8, buffer=self._shm.buf
        )
        self._owned = True

    @classmethod
    def attach(
        cls,
        shm_name: str,
        n_slots: int,
        n_bands: int,
        sig_dtype: np.dtype,
        slot_size: int,
        effective_bytes: int,
        locks: list[multiprocessing.synchronize.Lock],
        hash_strategy: HashStrategy | None = None,
    ) -> "SharedDirectMappedCache":
        obj = object.__new__(cls)
        obj._hash = hash_strategy or Blake2bHashStrategy()
        obj.n_bands = n_bands
        obj.sig_dtype = np.dtype(sig_dtype)
        obj.sig_size = n_bands * obj.sig_dtype.itemsize
        obj.slot_size = slot_size
        obj.n_slots = n_slots
        obj.effective_bytes = effective_bytes
        obj.shm_name = shm_name
        obj.locks = locks
        obj.n_locks = len(locks)
        obj._shm = shared_memory.SharedMemory(create=False, name=shm_name)
        obj._buf = np.ndarray(effective_bytes, dtype=np.uint8, buffer=obj._shm.buf)
        obj._owned = False
        return obj

    @property
    def enabled(self) -> bool:
        return True

    def _slot_offset(self, slot_id: int) -> int:
        return slot_id * self.slot_size

    def _get_slot_state(self, buf: np.ndarray, slot_id: int) -> int:
        return int(buf[self._slot_offset(slot_id)])

    def _get_slot_sig(self, buf: np.ndarray, slot_id: int) -> bytes:
        off = self._slot_offset(slot_id) + 1
        return bytes(buf[off : off + self.sig_size])

    def _get_slot_value(self, buf: np.ndarray, slot_id: int) -> int:
        return int(buf[self._slot_offset(slot_id) + 1 + self.sig_size])

    def _write_slot(self, buf: np.ndarray, slot_id: int, sig_bytes: bytes, value: int) -> None:
        off = self._slot_offset(slot_id)
        buf[off] = 1
        buf[off + 1 : off + 1 + self.sig_size] = np.frombuffer(sig_bytes, dtype=np.uint8)
        buf[off + 1 + self.sig_size] = value

    def lookup_many(self, key_array: np.ndarray) -> tuple[np.ndarray, np.ndarray, LookupStats]:
        n = len(key_array)
        out_values = np.zeros(n, dtype=np.uint8)
        hit_mask = np.zeros(n, dtype=bool)
        stats = LookupStats(n_queries=n)

        buf = self._buf
        sig_size = self.sig_size
        slot_size = self.slot_size

        for i in range(n):
            sig_bytes = key_array[i].tobytes()
            slot_id = self._hash.hash(sig_bytes) % self.n_slots
            lock_id = slot_id % self.n_locks

            with self.locks[lock_id]:
                off = slot_id * slot_size
                state = buf[off]
                if state == 0:
                    stats.n_misses_empty += 1
                else:
                    stored_sig = bytes(buf[off + 1 : off + 1 + sig_size])
                    if stored_sig == sig_bytes:
                        out_values[i] = buf[off + 1 + sig_size]
                        hit_mask[i] = True
                        stats.n_hits += 1
                    else:
                        stats.n_misses_collision += 1

        return out_values, hit_mask, stats

    def insert_many(self, key_array: np.ndarray, values: np.ndarray) -> InsertStats:
        n = len(key_array)
        stats = InsertStats(n_attempts=n)
        buf = self._buf
        sig_size = self.sig_size
        slot_size = self.slot_size

        for i in range(n):
            sig_bytes = key_array[i].tobytes()
            slot_id = self._hash.hash(sig_bytes) % self.n_slots
            lock_id = slot_id % self.n_locks

            with self.locks[lock_id]:
                off = slot_id * slot_size
                state = buf[off]
                if state == 0:
                    buf[off] = 1
                    buf[off + 1 : off + 1 + sig_size] = np.frombuffer(
                        sig_bytes, dtype=np.uint8
                    )
                    buf[off + 1 + sig_size] = values[i]
                    stats.n_inserted += 1
                else:
                    stored_sig = bytes(buf[off + 1 : off + 1 + sig_size])
                    if stored_sig != sig_bytes:
                        stats.n_skipped_collision += 1

        return stats

    def occupancy(self) -> float:
        sample_size = min(self.n_slots, 10000)
        indices = np.linspace(0, self.n_slots - 1, sample_size, dtype=int)
        filled = sum(1 for i in indices if self._buf[i * self.slot_size] == 1)
        return filled / sample_size

    def info(self) -> dict[str, str | int | float]:
        return {
            "cache_backend": "SharedDirectMappedCache",
            "n_bands": self.n_bands,
            "sig_dtype": str(self.sig_dtype),
            "slot_size_bytes": self.slot_size,
            "effective_slot_count": self.n_slots,
            "effective_cache_bytes": self.effective_bytes,
            "effective_cache_mb": round(self.effective_bytes / 1024 / 1024, 4),
            "n_locks": self.n_locks,
        }

    def close(self) -> None:
        try:
            self._shm.close()
            if getattr(self, "_owned", True):
                self._shm.unlink()
        except Exception:
            pass

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
