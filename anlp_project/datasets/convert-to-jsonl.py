from anlp_project.datasets.europarl import EuroParlRaw


def main():
    dataset = EuroParlRaw()
    with open('dataset.jsonl', 'w') as f:
        for i in range(len(dataset)):
            de, en = dataset[i]
            f.write(f'{{ "translation": {{ "en": "{en}", "de": "{de}" }} }}\n')


if __name__ == "__main__":
    main()