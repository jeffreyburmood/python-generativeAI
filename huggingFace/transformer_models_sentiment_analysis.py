""" This file contains code samples for using transformer sentiment analysis models the Hugging Face LLM course. """

from transformers import pipeline

# let the pipeline handle all of the steps
#
# Specify the model and version
# model_name = "distilbert-base-uncased-finetuned-sst-2-english"
# sentiment_pipeline = pipeline("sentiment-analysis", model=model_name)
#
# # Use the pipeline
# result = sentiment_pipeline(
#     ["I love the new Hugging Face models!", "I hate when they do not update the lessons!"]
# )
# print(result)

# perform the pipeline steps independently
# Tokenizer step
from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate when lessons are not updated!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(inputs)

# Model step
from transformers import AutoModel

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModel.from_pretrained(checkpoint)

outputs = model(**inputs)
print(outputs.last_hidden_state.shape)


# import torch
# from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
#
# tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
# model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
#
# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# with torch.no_grad():
#     logits = model(**inputs).logits
#
# predicted_class_id = logits.argmax().item()
# print(model.config.id2label[predicted_class_id])

