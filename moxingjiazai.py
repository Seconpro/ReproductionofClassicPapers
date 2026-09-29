from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

model_path = "./biobert-ner-final"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)

text = "The patient was diagnosed with lung cancer ."
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, is_split_into_words=False)
outputs = model(**inputs)
predictions = torch.argmax(outputs.logits, dim=-1)
id2label = {0:'O', 1:'B-Disease', 2:'I-Disease'}
pred_labels = [id2label[p.item()] for p in predictions[0]]
print(list(zip(tokenizer.tokenize(text), pred_labels)))