from collections import Counter
import logging
import multiprocessing
from pathlib import Path
import sqlite3
from typing import Tuple
import pickle
from os import path

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
        self._conn = sqlite3.connect(db_path)
        self.len = int(
            self._conn.cursor().execute("select count(*) from dataset").fetchone()[0]
        )

    def __getitem__(self, idx):
        row = (
            self._conn.cursor()
            .execute("select * from dataset where rowid = (?)", (idx + 1,))
            .fetchone()
        )
        return row

    def __len__(self) -> int:
        return self.len


class EuroParl(EuroParlRaw):
    def __init__(
        self,
        config: Config = Config.from_file(),
        db_path: Path = DATA_ROOT / "dataset.sqlite",
        load_from_pickle=False
    ):
        super().__init__(db_path=db_path)

        self.config = config
        self.de_tok = MosesTokenizer(lang="de")
        self.en_tok = MosesTokenizer()

        if not load_from_pickle:
            self.w2i_de, self.w2i_en = self.prepare_mappings(db_path)
        else:
            assert path.isfile(self.config.pickle_path), \
                "Please first pass load_from_pickle=False to gnenerate the pickle file"
            pickle_file = open(self.config.pickle_path, 'rb')
            self.w2i_de, self.w2i_en = pickle.load(pickle_file)
            pickle_file.close()

        self.de_vocab_size = len(self.w2i_de)
        self.en_vocab_size = len(self.w2i_en)

    # TODO: why is db_path asked for here when it was also taken in the __init__ method?
    def prepare_mappings(self, db_path: Path):
        total_size = self.len
        n_procs = multiprocessing.cpu_count()
        chunk_size = total_size // n_procs

        logging.info('total_size: %d, n_procs: %d, chunk_size: %d', total_size, n_procs, chunk_size)

        # break up indices to be equally consumed by all processes
        # stores [start, end)
        proc_indices = [i * chunk_size for i in range(n_procs)]
        proc_indices.append(total_size)

        # we are showing progress bar for worker 0
        # reverse the list so that worker 0 starts/ends the last
        args = reversed([
            (db_path, proc_indices[i], proc_indices[i + 1])
            for i in range(len(proc_indices) - 1)
        ])

        # unk index will be 0
        wf_de, wf_en = Counter(), Counter()

        with multiprocessing.Pool(n_procs) as pool:
            local_wfs = pool.starmap(_word_freq_calculator, args)

        for local_wf_de, local_wf_en in tqdm(
            local_wfs, desc="Aggregating word frequencies"
        ):
            wf_de += local_wf_de
            wf_en += local_wf_en

        w2i_de = {"__UNKNOWN__": 0}
        w2i_en = {"__UNKNOWN__": 0}

        i = 1
        for w, f in tqdm(wf_de.items(), desc="Mapping German words to indices"):
            if f >= self.config.min_occurances_for_vocab:
                w2i_de[w] = i
                i += 1

        i = 1
        for w, f in tqdm(wf_en.items(), desc="Mapping English words to indices"):
            if f >= self.config.min_occurances_for_vocab:
                w2i_en[w] = i
                i += 1

        write_file = open(self.config.pickle_path, 'wb')
        pickle.dump((w2i_de, w2i_en), write_file)
        write_file.close()
        return w2i_de, w2i_en

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        row = super().__getitem__(idx)

        de = [self.w2i_de.get(token, 0) for token in self.de_tok.tokenize(row[0])]
        en = [self.w2i_en.get(token, 0) for token in self.en_tok.tokenize(row[1])]

        assert max(len(de), len(en)) <= self.config.max_length, "You need to raise max length"
        de_1hot = np.zeros((self.config.max_length, self.de_vocab_size), dtype=int)
        de_1hot[np.arange(len(de)), de] = 1

        en_1hot = np.zeros((self.config.max_length, self.en_vocab_size), dtype=int)
        en_1hot[np.arange(len(en)), en] = 1

        return de_1hot, en_1hot
