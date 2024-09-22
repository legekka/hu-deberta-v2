In this project we aim for creating a new hungarian BERT language model. We are using Huggingface libraries to achieve it, CulturaX hu dataset for training.

## hu-deBERTa-v2
The model is based on Microsoft's DeBERTa-V2 architecture. We are aiming for a larger context size than other available hungarian BERT models, but first we pretrained it for 512 tokens.

## Evaluation
After the pretraining we use the HuLU metrics for evaluation:
- HuCOLA
- HuSST