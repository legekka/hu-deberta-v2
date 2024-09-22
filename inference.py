import argparse

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', type=str, required=True, help='Huggingface model name')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForMaskedLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    model.to(device)
    model.eval()

    text = "Nyelvi modellek készítésekor [MASK] a megfelelő diverzitású tananyag használata."


    inputs = tokenizer(text, return_tensors="pt")
    inputs.to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        # order the tokens by their probability, and get the top 5 with their probabilities
        probs = logits.softmax(dim=-1)[0, inputs['input_ids'][0].tolist().index(tokenizer.mask_token_id)].topk(5)
        for token, prob in zip(tokenizer.convert_ids_to_tokens(probs.indices), probs.values):
            print(f"{token}: {prob:.5f}")
