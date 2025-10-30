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
# There are many different architectures available in 🤗 Transformers, with each one designed around tackling a
# specific task. Here is a non-exhaustive list:
#
# *Model (retrieve the hidden states)
# *ForCausalLM
# *ForMaskedLM
# *ForMultipleChoice
# *ForQuestionAnswering
# *ForSequenceClassification
# *ForTokenClassification
# and others 🤗

# we will need a model with a sequence classification head (to be able to classify the sentences as positive or negative).
# So, we won’t actually use the AutoModel class, but AutoModelForSequenceClassification:

# from transformers import AutoModel
#
# checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
# model = AutoModel.from_pretrained(checkpoint)
#
# outputs = model(**inputs)
# print(outputs.last_hidden_state.shape)

from transformers import AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**inputs)

# Since we have just two sentences and two labels, the result we get from our model is of shape 2 x 2.
print(outputs.logits.shape)

# post processing the model output
print(outputs.logits)

# Our model predicted [-1.5607, 1.6123] for the first sentence and [ 4.1692, -3.3464] for the second one. Those are not
# probabilities but logits, the raw, unnormalized scores outputted by the last layer of the model. To be converted to
# probabilities, they need to go through a SoftMax layer (all 🤗 Transformers models output the logits, as the loss
# function for training will generally fuse the last activation function, such as SoftMax, with the actual loss function,
# such as cross entropy):

import torch

predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)

# Now we can see that the model predicted [0.0402, 0.9598] for the first sentence and [0.9995, 0.0005] for the second
# one. These are recognizable probability scores.
#
# To get the labels corresponding to each position, we can inspect the id2label attribute of the model config

print(model.config.id2label)

# Now we can conclude that the model predicted the following:
#
# First sentence: NEGATIVE: 0.0402, POSITIVE: 0.9598
# Second sentence: NEGATIVE: 0.9995, POSITIVE: 0.0005

# We have successfully reproduced the three steps of the pipeline: preprocessing with tokenizers, passing the inputs
# through the model, and postprocessing!




# TypingMind example
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

