import hashlib
import struct
from pathlib import Path

import lmdb
import msgpack
import numpy as np
import torch
from radfinder.utils.logging_utils import log_info

from visiontext.iotools.feature_compression import convert_to_fp16_torch


class TextFeatureStore:
    """
    Store text embeddings and texts in LMDB databases.

    For qwen3 hidden_dim=1024, context_length=1024, storagetype fp16: 2MB per datapoint.
    """

    def __init__(self, text_features_dir: Path, map_size: int = 1024**4) -> None:
        text_features_dir = Path(text_features_dir)
        self.text_features_dir = text_features_dir
        self.map_size = map_size
        self._embeddings_db: lmdb.Environment | None = None
        self._texts_db: lmdb.Environment | None = None
        self._pool_db: lmdb.Environment | None = None
        self._read_emb_txn: lmdb.Transaction | None = None
        self._read_txt_txn: lmdb.Transaction | None = None
        self._read_pool_txn: lmdb.Transaction | None = None
        self._write_emb_txn: lmdb.Transaction | None = None
        self._write_txt_txn: lmdb.Transaction | None = None
        self._write_pool_txn: lmdb.Transaction | None = None

    @property
    def embeddings_db(self) -> lmdb.Environment:
        if self._embeddings_db is None:
            self._embeddings_db = self._open_lmdb(
                self.text_features_dir, "embeddings.lmdb", self.map_size
            )
        return self._embeddings_db

    @property
    def texts_db(self) -> lmdb.Environment:
        if self._texts_db is None:
            self._texts_db = self._open_lmdb(self.text_features_dir, "texts.lmdb", self.map_size)
        return self._texts_db

    @property
    def pool_db(self) -> lmdb.Environment:
        if self._pool_db is None:
            self._pool_db = self._open_lmdb(self.text_features_dir, "pool.lmdb", self.map_size)
        return self._pool_db

    def keys(self) -> list[bytes]:
        assert self._read_emb_txn is not None, "Use open_readonly() before reading embeddings"
        return list(self._read_emb_txn.cursor().iternext(keys=True, values=False))

    def values(self) -> list[bytes]:
        assert self._read_emb_txn is not None, "Use open_readonly() before reading embeddings"
        return list(self._read_emb_txn.cursor().iternext(keys=False, values=True))

    def items(self) -> list[tuple[bytes, bytes]]:
        assert self._read_emb_txn is not None, "Use open_readonly() before reading embeddings"
        return list(self._read_emb_txn.cursor().iternext(keys=True, values=True))

    @staticmethod
    def _open_lmdb(text_features_dir: Path, db_name: str, map_size: int) -> lmdb.Environment:
        lmdb_path = text_features_dir / db_name
        return lmdb.open(lmdb_path.as_posix(), map_size=map_size)

    @staticmethod
    def get_text_hash(text: str) -> bytes:
        return hashlib.blake2b(text.encode("utf-8"), digest_size=32).digest()

    def open_readonly(self) -> None:
        self.close_readonly()
        self._read_emb_txn = self.embeddings_db.begin(write=False)
        self._read_txt_txn = self.texts_db.begin(write=False)
        self._read_pool_txn = self.pool_db.begin(write=False)

    def open_writeable(self) -> None:
        self.close_writeable()
        self._write_emb_txn = self.embeddings_db.begin(write=True)
        self._write_txt_txn = self.texts_db.begin(write=True)
        self._write_pool_txn = self.pool_db.begin(write=True)

    def commit(self) -> None:
        if self._write_emb_txn is not None:
            self._write_emb_txn.commit()
            self._write_emb_txn = self.embeddings_db.begin(write=True)
        if self._write_txt_txn is not None:
            self._write_txt_txn.commit()
            self._write_txt_txn = self.texts_db.begin(write=True)
        if self._write_pool_txn is not None:
            self._write_pool_txn.commit()
            self._write_pool_txn = self.pool_db.begin(write=True)

    @staticmethod
    def _abort_ignore_error(txn: lmdb.Transaction) -> None:
        try:
            txn.abort()
        except Exception:
            pass

    def close_readonly(self) -> None:
        if self._read_emb_txn is not None:
            self._abort_ignore_error(self._read_emb_txn)
            self._read_emb_txn = None
        if self._read_txt_txn is not None:
            self._abort_ignore_error(self._read_txt_txn)
            self._read_txt_txn = None
        if self._read_pool_txn is not None:
            self._abort_ignore_error(self._read_pool_txn)
            self._read_pool_txn = None

    def close_writeable(self) -> None:
        if self._write_emb_txn is not None:
            self._abort_ignore_error(self._write_emb_txn)
            self._write_emb_txn = None
        if self._write_txt_txn is not None:
            self._abort_ignore_error(self._write_txt_txn)
            self._write_txt_txn = None
        if self._write_pool_txn is not None:
            self._abort_ignore_error(self._write_pool_txn)
            self._write_pool_txn = None

    def close(self) -> None:
        self.close_readonly()
        self.close_writeable()

    def has(self, text_hash: bytes) -> bool:
        assert self._read_emb_txn is not None, "Use open_readonly() before reading embeddings"
        return self._read_emb_txn.get(text_hash) is not None

    def put(
        self,
        text_hash: bytes,
        embeddings: torch.Tensor,
        metadata: dict,
    ) -> None:
        assert self._write_txt_txn is not None, "Use open_writeable() before writing texts"
        assert self._write_emb_txn is not None, "Use open_writeable() before writing embeddings"
        assert self._write_pool_txn is not None, "Use open_writeable() before writing pool"
        # create text payload
        texts_payload = msgpack.packb(metadata, use_bin_type=True)
        # create embeddings payload
        embeddings = convert_to_fp16_torch(embeddings).detach().contiguous().cpu()
        n_tokens, emb_dim = embeddings.shape
        raw_bytes = embeddings.numpy().tobytes()
        embeddings_payload = struct.pack("<II", n_tokens, emb_dim) + raw_bytes
        # create pooled embedding payload
        pooled_embedding = embeddings[-1]
        pooled_bytes = pooled_embedding.numpy().tobytes()
        pooled_payload = struct.pack("<I", emb_dim) + pooled_bytes
        # write to databases
        self._write_txt_txn.put(text_hash, texts_payload)
        self._write_emb_txn.put(text_hash, embeddings_payload)
        self._write_pool_txn.put(text_hash, pooled_payload)

    def get_embedding(self, text_hash: bytes) -> torch.Tensor | None:
        assert self._read_emb_txn is not None, "Use open_readonly() before reading embeddings"
        payload = self._read_emb_txn.get(text_hash)
        if payload is None:
            return None
        n_tokens, emb_dim = struct.unpack("<II", payload[:8])
        embeddings = np.frombuffer(payload, offset=8, dtype=np.float16, count=n_tokens * emb_dim)
        # we need to copy to avoid the non-writeable tensor warning, eventually it would need to
        # get copied to the GPU anyway, so it's safest to just copy here
        return torch.from_numpy(embeddings.reshape(n_tokens, emb_dim).copy())

    def get_pooled(self, text_hash: bytes) -> torch.Tensor | None:
        assert self._read_pool_txn is not None, "Use open_readonly() before reading pool"
        payload = self._read_pool_txn.get(text_hash)
        if payload is None:
            return None
        (emb_dim,) = struct.unpack("<I", payload[:4])
        pooled_embedding = np.frombuffer(payload, offset=4, dtype=np.float16, count=emb_dim)
        return torch.from_numpy(pooled_embedding.copy())

    def get_text(self, text_hash: bytes) -> tuple[str, list[int]] | None:
        assert self._read_txt_txn is not None, "Use open_readonly() before reading texts"
        payload = self._read_txt_txn.get(text_hash)
        if payload is None:
            return None
        text_dict = msgpack.unpackb(payload, raw=False)
        return text_dict["text"], text_dict["input_ids"]

    def check_db_sync(self) -> dict[str, int]:
        """Keep only keys that exist in all three DBs."""
        self.close()

        with (
            self.embeddings_db.begin(write=False) as emb_rtxn,
            self.texts_db.begin(write=False) as txt_rtxn,
            self.pool_db.begin(write=False) as pool_rtxn,
        ):
            emb_keys = set(emb_rtxn.cursor().iternext(keys=True, values=False))
            txt_keys = set(txt_rtxn.cursor().iternext(keys=True, values=False))
            pool_keys = set(pool_rtxn.cursor().iternext(keys=True, values=False))

        union_keys = emb_keys | txt_keys | pool_keys
        cut_keys = emb_keys & txt_keys & pool_keys
        to_delete = union_keys - cut_keys

        if not to_delete:
            return

        deleted_counts = {}
        with self.embeddings_db.begin(write=True) as emb_wtxn:
            deleted_counts["embeddings"] = sum(emb_wtxn.delete(k) for k in to_delete)
        with self.texts_db.begin(write=True) as txt_wtxn:
            deleted_counts["texts"] = sum(txt_wtxn.delete(k) for k in to_delete)
        with self.pool_db.begin(write=True) as pool_wtxn:
            deleted_counts["pool"] = sum(pool_wtxn.delete(k) for k in to_delete)

        deleted_counts_str = ", ".join(
            f"{name}={count}" for name, count in sorted(deleted_counts.items())
        )
        log_info(f"Deleted out-of-sync keys: {deleted_counts_str}")

    def update_pooled(self) -> None:
        """Create pooled embeddings for missing keys in pool.lmdb."""
        self.close()
        self.open_readonly()
        emb_keys = set(self._read_emb_txn.cursor().iternext(keys=True, values=False))
        pool_keys = set(self._read_pool_txn.cursor().iternext(keys=True, values=False))
        missing_keys = emb_keys - pool_keys
        self.close_readonly()

        if not missing_keys:
            log_info("No missing pooled embeddings found")
            return
        log_info(f"Writing pooled embeddings for {len(missing_keys)} keys")
        self._read_emb_txn = self.embeddings_db.begin(write=False)
        self._write_pool_txn = self.pool_db.begin(write=True)

        for text_hash in missing_keys:
            embeddings = self.get_embedding(text_hash)
            assert embeddings is not None, f"Missing embeddings for {text_hash=}"
            pooled_embedding = embeddings[-1].detach().contiguous().cpu()
            pooled_bytes = pooled_embedding.numpy().tobytes()
            emb_dim = pooled_embedding.shape[0]
            pooled_payload = struct.pack("<I", emb_dim) + pooled_bytes
            self._write_pool_txn.put(text_hash, pooled_payload)
        self.commit()

        self._write_pool_txn.commit()
        self._write_pool_txn = None
        self._abort_ignore_error(self._read_emb_txn)
        self._read_emb_txn = None

        log_info(f"Added pooled embeddings for {len(missing_keys)} entries")
