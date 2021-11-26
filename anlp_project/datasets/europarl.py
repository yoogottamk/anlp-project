import copy
import logging
import multiprocessing
import pickle
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Tuple, List

import numpy as np
from sacremoses.tokenize import MosesTokenizer
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from anlp_project.config import DATA_ROOT, Config


def _word_freq_calculator(db_path, start, end):
    raw_ds = EuroParlRaw(db_path)

    de_tok, en_tok = MosesTokenizer(lang="de"), MosesTokenizer()
    wf_de, wf_en = Counter(), Counter()

    iterator = range(start, end)
    is_first_worker = start == 0
    if is_first_worker:
        iterator = tqdm(iterator, desc="[w0] Calculating word frequencies")

    for idx in iterator:
        de, en = raw_ds[idx]

        for token in de_tok.tokenize(de):
            wf_de[token] += 1
        for token in en_tok.tokenize(en):
            wf_en[token] += 1

    return wf_de, wf_en


class EuroParlRaw(Dataset):
    def __init__(self, db_path=DATA_ROOT / "dataset.sqlite"):
        super().__init__()
        self.db_path = db_path
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._len = int(
            conn.cursor().execute("select count(*) from dataset").fetchone()[0]
        )

    def __getitem__(self, idx):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        row = (
            conn.cursor()
            .execute("select * from dataset where rowid = (?)", (idx + 1,))
            .fetchone()
        )
        return row

    def __len__(self) -> int:
        return self._len


class EuroParl(EuroParlRaw):
    UNK_TOKEN_INDEX = 0
    BOS_TOKEN_INDEX = 1
    EOS_TOKEN_INDEX = 2

    def __init__(
        self,
        config: Config = Config.from_file(),
        db_path: Path = DATA_ROOT / "dataset.sqlite",
        force_regenerate_mappings=False,
    ):
        super().__init__(db_path=db_path)

        self.config = config
        self._len = int(self._len * self.config.dataset_fraction)

        # bad hack for making it work
        # always remain a multiple of 4
        self._len -= self._len % 4

        self.de_tok = MosesTokenizer(lang="de")
        self.en_tok = MosesTokenizer()

        if force_regenerate_mappings or not Path(self.config.pickle_path).is_file():
            self.w2i_de, self.w2i_en = self.prepare_mappings(db_path)
        else:
            with open(self.config.pickle_path, "rb") as pickle_file:
                self.w2i_de, self.w2i_en = pickle.load(pickle_file)

        self.de_vocab_size = len(self.w2i_de)
        self.en_vocab_size = len(self.w2i_en)

    # TODO: why is db_path asked for here when it was also taken in the __init__ method?
    def prepare_mappings(self, db_path: Path):
        total_size = len(self)
        n_procs = multiprocessing.cpu_count()
        chunk_size = total_size // n_procs

        logging.info(
            "total_size: %d, n_procs: %d, chunk_size: %d",
            total_size,
            n_procs,
            chunk_size,
        )

        # break up indices to be equally consumed by all processes
        # stores [start, end)
        proc_indices = [i * chunk_size for i in range(n_procs)]
        proc_indices.append(total_size)

        # we are showing progress bar for worker 0
        # reverse the list so that worker 0 starts/ends the last
        args = reversed(
            [
                (db_path, proc_indices[i], proc_indices[i + 1])
                for i in range(len(proc_indices) - 1)
            ]
        )

        # unk index will be 0
        wf_de, wf_en = Counter(), Counter()

        with multiprocessing.Pool(n_procs) as pool:
            local_wfs = pool.starmap(_word_freq_calculator, args)

        for local_wf_de, local_wf_en in tqdm(
            local_wfs, desc="Aggregating word frequencies"
        ):
            wf_de += local_wf_de
            wf_en += local_wf_en

        w2i_de = {
            "__PAD__": 0,
            "__UNKNOWN__": 1,
            self.config.bos_token: 2,
            self.config.eos_token: 3,
        }
        w2i_en = copy.deepcopy(w2i_de)

        i = len(w2i_de)
        for w, f in tqdm(wf_de.items(), desc="Mapping German words to indices"):
            if f >= self.config.min_occurances_for_vocab:
                w2i_de[w] = i
                i += 1

        i = len(w2i_en)
        for w, f in tqdm(wf_en.items(), desc="Mapping English words to indices"):
            if f >= self.config.min_occurances_for_vocab:
                w2i_en[w] = i
                i += 1

        Path(self.config.pickle_path).parent.mkdir(exist_ok=True, parents=True)
        with open(self.config.pickle_path, "wb") as write_file:
            pickle.dump((w2i_de, w2i_en), write_file)
        return w2i_de, w2i_en

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        row = super().__getitem__(idx)
        assert row, len(self)

        de_tokens = [
            self.config.bos_token,
            *self.de_tok.tokenize(row[0]),
        ][: self.config.max_length - 1]
        en_tokens = [
            self.config.bos_token,
            *self.en_tok.tokenize(row[1]),
        ][: self.config.max_length - 1]
        de_tokens.append(self.config.eos_token)
        en_tokens.append(self.config.eos_token)

        # TODO: fix this in preprocessing
        # lower part not part of TODO
        # Make UNK 1 for pad and UNK to be diff symbols
        de = np.array([self.w2i_de.get(token, 1) for token in de_tokens])
        en = np.array([self.w2i_en.get(token, 1) for token in en_tokens])

        padded_de = np.pad(de, (0, self.config.max_length - len(de)))
        padded_en = np.pad(en, (0, self.config.max_length - len(en)))

        return padded_de, padded_en

    def sentence_to_indices(self, sentence: str):
        tokens = self.de_tok.tokenize(sentence)
        indices = [self.w2i_de.get(token, 0) for token in tokens]
        return indices

    def indices_to_sentence(self, indices: List[int]):
        tokens = []
        # TODO: pickle the reverse lookup object also
        for idx in indices:
            for k, v in self.w2i_en.items():
                if v == idx:
                    tokens.append(k)

        return " ".join(tokens)
