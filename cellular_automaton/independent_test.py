import json
from pathlib import Path
from types import SimpleNamespace

import torch
from models.lstm_ut_bidir import GPTBase

CHECKPOINT = Path(
    "/iridisfs/scratch/ab3u21/exps/cellular-automaton/rule30/"
    "lstm_ut_bidir/"
    "ca_r4_r6_first_ood_v1_r6_lstm_learned_previous_and_proposed_zero_seed0/"
    "best_id.pt"
)

MANIFEST = Path(
    "/iridisfs/scratch/tak1e25/CoTFormer-private/"
    "iridis/ca-rule30/runs/run_34__ca_lstm_r6_zero/run_manifest.json"
)

INITIAL = "1011001001101010010111010001101011000101011010010011101001010010"
CA_STEPS = 20       # Desired Rule 30 horizon.
NUM_REPEATS = 20    # Internal model repeats; choose before seeing the result.

# Restore the architecture and every saved model weight.
args = SimpleNamespace(**json.loads(MANIFEST.read_text())["resolved_args"])
assert args.model == "lstm_ut_bidir"
assert args.vocab_size == 2
assert set(INITIAL) <= {"0", "1"}

model = GPTBase(args)
checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
model.load_state_dict(checkpoint["model"], strict=True)
model.eval()

# The model API requires [1, number_of_cells]: exactly one example.
inputs = torch.tensor([list(map(int, INITIAL))], dtype=torch.long)

with torch.inference_mode():
    logits = model(
        inputs,
        num_repeats=NUM_REPEATS,
        get_logits=True,
        return_all_logits=True,
        use_cache=False,
    )["logits"]

assert logits.shape == (1, len(INITIAL), 2)
prediction = "".join(map(str, logits[0].argmax(dim=-1).tolist()))

# Independent Rule 30 truth calculation, AFTER prediction.
# Table indexed by 4*left + 2*centre + right; periodic boundaries.
rule30 = (0, 1, 1, 1, 1, 0, 0, 0)
truth = list(map(int, INITIAL))
for _ in range(CA_STEPS):
    truth = [
        rule30[4 * truth[i - 1] + 2 * truth[i] + truth[(i + 1) % len(truth)]]
        for i in range(len(truth))
    ]
expected = "".join(map(str, truth))
correct = sum(a == b for a, b in zip(prediction, expected))

result = (
    f"Checkpoint: {CHECKPOINT}\n"
    f"Checkpoint iteration: {checkpoint['itr']}\n"
    f"Manifest: {MANIFEST}\n"
    f"Training (CA steps, repeats): {args.ca_train_pairs}\n"
    f"CA steps: {CA_STEPS}; model repeats: {NUM_REPEATS}\n"
    f"Input:      {INITIAL}\n"
    f"Prediction: {prediction}\n"
    f"Expected:   {expected}\n"
    f"Correct cells: {correct}/{len(INITIAL)}\n"
    f"Exact match: {prediction == expected}\n"
)
OUTPUT.write_text(result)
print(result, end="")