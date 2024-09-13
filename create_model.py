from transformers import DebertaV2ForMaskedLM, DebertaV2Config, AutoTokenizer

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("tokenizer", use_fast=True)

config = DebertaV2Config(
    attention_probs_dropout_prob=0.1,
    hidden_act="gelu",
    hidden_dropout_prob=0.1,
    hidden_size=768,
    initializer_range=0.02,
    intermediate_size=3072,
    max_position_embeddings=2048,
    relative_attention=True,
    pos_att_type="c2p|p2c",
    layer_norm_eps=1e-7,
    max_relative_positions=-1,
    position_biased_input=False,
    num_attention_heads=12,
    num_hidden_layers=12,
    type_vocab_size=0,
    vocab_size=tokenizer.vocab_size,
    pad_token_id=tokenizer.pad_token_id
)

model = DebertaV2ForMaskedLM(config)
# print model paramter count
print(f"Model initialized with {model.num_parameters()} parameters.")
# print tokenizer vocab size
print(f"Tokenizer initialized with {len(tokenizer)} tokens.")

# model name: hu-deberta-v2-base
model.save_pretrained("hu-deberta-v2-base")
tokenizer.save_pretrained("hu-deberta-v2-base")