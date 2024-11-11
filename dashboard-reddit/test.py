from transformers import BertForSequenceClassification, BertTokenizer
import torch
from shared import app_dir

model_dir = app_dir / 'fine_tuned_bert_model_3'
model = BertForSequenceClassification.from_pretrained(str(model_dir), output_attentions=True)
tokenizer = BertTokenizer.from_pretrained(str(model_dir))
model.eval()
print(model)
