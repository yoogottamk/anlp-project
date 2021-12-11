import sys

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm.auto import tqdm
from sacrebleu.metrics import BLEU

from anlp_project.datasets.europarl import EuroParlRaw

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} CHECKPOINT_PATH")
    sys.exit(1)

checkpoint_name = sys.argv[1]

if "en-de" in checkpoint_name:
    prompt = "translate from English to German: "
    source_idx = 1
elif "de-en" not in checkpoint_name:
    prompt = "translate from German to English: "
    source_idx = 0
else:
    print("Checkpoint name doesn't have en-de or de-en. Can't figure out direction")
    sys.exit(1)

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("/scratch/en-de/")

# Initialize the model
model = AutoModelForSeq2SeqLM.from_pretrained("/scratch/en-de/")
bleu = BLEU(lowercase=True)

ds = EuroParlRaw()
N = len(ds)
batch_size = 100

score = 0
refs = [[]]
hyps = []
for i in tqdm(range(N // batch_size)):
    batch = []
    for x in range(i * batch_size, (i + 1) * batch_size):
        batch.append(f"{prompt}: {ds[x][source_idx]}")
        refs[0].append(ds[x][1 - source_idx])
    tokenized_text = tokenizer(
        batch, padding="max_length", truncation=True, return_tensors="pt"
    )

    # Perform translation and decode the output
    translation = model.generate(**tokenized_text)
    translated_text = tokenizer.batch_decode(translation, skip_special_tokens=True)

    hyps += translated_text

for i in range(len(hyps)):
    print(hyps[i], refs[0][i], sep="\n")
    print("---------------------------")

print(bleu.corpus_score(hyps, refs))
