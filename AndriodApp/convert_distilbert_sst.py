from transformers.utils.dummy_pt_objects import DistilBertForSequenceClassification
import torch
from torch.utils.mobile_optimizer import optimize_for_mobile
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

model.eval()

text = "I love you."
# inputs['input_ids'].size() is 360, the maximum size of the input tokens generated from the user question and text
# on mobile apps, if the size of the input tokens of the text and question is less than 360, padding will be needed to make the model work correctly.

inputs = tokenizer(text, return_tensors='pt')
#model_dynamic_quantized = torch.quantization.quantize_dynamic(model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint16)
traced_model = torch.jit.trace(model, inputs['input_ids'], strict=False)
optimized_traced_model = optimize_for_mobile(traced_model)
optimized_traced_model._save_for_lite_interpreter("sst_quantized.ptl")
# 360 is the length of model input, i.e. the length of the tokenized ids of question+text