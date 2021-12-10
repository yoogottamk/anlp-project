#/usr/bin/python3

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
from tqdm import tqdm

filepath = "dataset/intermediate/de-en.tsv"

if not os.path.exists(filepath):
    print("Please run this script from root project directory. Also make you already have downloaded the full dataset using the scripts we have provided")
    exit(1)

try:
    languages = [sys.argv[1]]
except:
    languages = ["de", "en"]

def make_plots():
    de_counts, en_counts = [], []

    invalid_sentence_count = 0
    valid_sentence_count = 0

    print("Reading through the file, should take 5-10 seconds")

    with open(filepath, "r") as f:
        for line in tqdm(f):
            try:
                de, en = line.split("\t")
                de_toks = de.split(" ")
                en_tokens = en.split(" " )
                de_counts.append(len(de_toks))
                en_counts.append(len(en_tokens))
                valid_sentence_count += 1
            except ValueError:
                invalid_sentence_count += 1

    token_diff = [de_counts[i] - en_counts[i] for i in range(len(de_counts))]
    token_diff.sort()
    token_series = pd.Series(token_diff, dtype=np.int64)

    de_counts.sort()
    en_counts.sort()
    de_series = pd.Series(de_counts, dtype=np.int64)
    en_series = pd.Series(en_counts, dtype=np.int64)

    line_colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'purple']

    # token_counts[0].sort()
    # token_counts[1].sort()
    print(f"Total sentences processed: {valid_sentence_count}; Sentences skipped: {invalid_sentence_count}")

    print("Statistics on the token counts")
    print("German")
    print(de_series.describe().astype(np.int64))
    print("English")
    print(en_series.describe().astype(np.int64))

    print("Token count differences")
    print(token_series.describe().astype(np.int64))
    print("Token difference 2% quantile", token_series.quantile(q=0.02))
    print("Token difference 10% quantile", token_series.quantile(q=0.1))
    print("Token difference 90% quantile", token_series.quantile(q=0.9))
    print("Token difference 97% quantile", token_series.quantile(q=0.97))
    plt.xlabel("Index of sentence pair")
    plt.ylabel("Difference value")
    plt.plot(token_series, label="German token count - English token count", color=line_colors[-1])
    plt.legend()
    plt.show()

    for language in languages:
        language_label = "English" if language == "en" else "German"

        plot_series = de_series if language == "de" else en_series
        plt.xlabel("Index of sentence pair")
        plt.ylabel("Token count")
        plt.plot(plot_series, label=f"{language_label} token counts", color=line_colors[0]) 

        for index, quantile in enumerate([0.5, 0.75, 0.95, 0.99, 0.9999]):
            if quantile > 0.99:
                quantile_label = quantile*100
            else:
                quantile_label = int(quantile*100)

            med = plot_series.quantile(q=quantile)
            plt.axhline(y=med, linestyle="dashed", label=f"{language_label} token count {quantile_label}% quantile", color=line_colors[index + 1])

        plt.legend()
        plt.show()


if __name__ == "__main__":
    make_plots()
