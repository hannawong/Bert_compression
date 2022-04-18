from transformers.utils.dummy_pt_objects import DistilBertForSequenceClassification
import torch
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
from torch.utils.mobile_optimizer import optimize_for_mobile
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/stsb-distilbert-base')
model = AutoModel.from_pretrained('sentence-transformers/stsb-distilbert-base')

text = "I love you."
text2= "I like you"

inputs = tokenizer(text,text2, return_tensors='pt')
#model_dynamic_quantized = torch.quantization.quantize_dynamic(model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint16)
traced_model = torch.jit.trace(model, inputs['input_ids'], strict=False)
optimized_traced_model = optimize_for_mobile(traced_model)
optimized_traced_model._save_for_lite_interpreter("sst_quantized.ptl")