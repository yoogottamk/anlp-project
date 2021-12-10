#/usr/bin/python3

from matplotlib import pyplot as plt
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

    de_counts.sort()
    en_counts.sort()
    de_series = pd.Series(de_counts, dtype=int)
    en_series = pd.Series(en_counts, dtype=int)

    # token_counts[0].sort()
    # token_counts[1].sort()
    print(f"Total sentences processed: {valid_sentence_count}; Sentences skipped: {invalid_sentence_count}")

    print("Statistics on the token counts")
    print("German")
    print(de_series.describe())
    print("English")
    print(en_series.describe())

    line_colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    for language in languages:
        language_label = "English" if language == "en" else "German"

        plot_series = de_series if language == "de" else en_series
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
