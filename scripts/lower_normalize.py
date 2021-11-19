import io
import multiprocessing
import re
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

from tqdm.auto import tqdm

fname = sys.argv[1]


def process_line(line: str):
    """
    There is a complex mess of stuff down there
    Use this function to do processing stuff to your line

    It is expected that the returned line ends with "\n"
    """
    text = line.strip().lower()
    no_accents = "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )
    no_quotes = re.sub(r"['\"(){}\[\]]", "", no_accents)

    return no_quotes + "\n"


def count_newlines(fname):
    """
    modified version of https://stackoverflow.com/a/68385697
    """
    count = 0

    with open(fname, "rb", buffering=0) as f:
        b = f.read(2 ** 16)
        while b:
            count += b.count(b"\n")
            b = f.read(2 ** 16)

    return count


def get_split_positions(n_procs, fname):
    """
    Find the file seek position for newline boundaries

    Almost equally split the file by newlines for independent
    processing by `n_procs` processes
    """
    total_lines = count_newlines(fname)
    per_proc_lines = total_lines // n_procs
    split_boundary = [i * per_proc_lines for i in range(1, n_procs)]
    end_scanning_at = []

    curr_count = 0
    with open(fname, "rb", buffering=0) as f:
        while len(split_boundary):
            buf = f.read(2 ** 10)
            if not buf:
                break
            curr_count += buf.count(b"\n")

            # we're very close to where we're supposed to split
            # read single bytes to find the next newline and add split there
            if curr_count > split_boundary[0]:
                byte_ = f.read(1)
                if byte_ == b"\n":
                    end_scanning_at.append(f.tell())
                    split_boundary.pop(0)

        f.seek(0, io.SEEK_END)
        end_scanning_at.append(f.tell())

    return [0, *end_scanning_at]


def process_file_segment(index, fname, start, end):
    last_pbar_refresh = time.time()
    with tqdm(total=(end - start), disable=(index != 0)) as pbar:
        with open(f"{fname}.shard{index:03d}", "w+") as wf:
            with open(fname, "r") as f:
                f.seek(start)

                while line := f.readline():
                    pbar.n = f.tell() - start
                    if (cur_time := time.time()) - last_pbar_refresh > 1:
                        pbar.refresh()
                        last_pbar_refresh = cur_time

                    wf.write(process_line(line))

                    if f.tell() > end:
                        break


def process_file(n_procs, fname):
    splits = get_split_positions(n_procs, fname)
    args = [(i, fname, splits[i], splits[i + 1] - 1) for i in range(n_procs)]

    with multiprocessing.Pool(n_procs) as pool:
        pool.starmap(process_file_segment, args)

    fname_path = Path(fname)
    shards = list(fname_path.parent.glob(f"{fname_path.name}.shard*"))
    shards.sort()

    with open(f"{fname_path.parent}/de-en-final.tsv", "w+") as f:
        subprocess.run(["cat"] + [str(f) for f in shards], stdout=f)


process_file(multiprocessing.cpu_count(), fname)
