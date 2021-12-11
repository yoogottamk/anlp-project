import json

from tqdm import tqdm

from anlp_project.datasets.europarl import EuroParlRaw


def main():
    """
    Converts dataset to jsonlines format

    needed for baseline++ training
    """
    dataset = EuroParlRaw()
    with open("dataset.jsonl", "w") as f:
        for de, en in tqdm(dataset):
            f.write(json.dumps({"translation": {"en": en, "de": de}}) + "\n")


if __name__ == "__main__":
    main()
