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
#
# Tokenizer step
#
# Translating text to numbers is known as encoding. Encoding is done in a two-step process: the tokenization, followed
# by the conversion to input IDs.
#
# As we’ve seen, the first step is to split the text into words (or parts of words, punctuation symbols, etc.),
# usually called tokens. There are multiple rules that can govern that process, which is why we need to instantiate
# the tokenizer using the name of the model, to make sure we use the same rules that were used when the model was pretrained.
#
# The second step is to convert those tokens into numbers, so we can build a tensor out of them and feed them to the model.
# To do this, the tokenizer has a vocabulary, which is the part we download when we instantiate it with the from_pretrained()
# method. Again, we need to use the same vocabulary used when the model was pretrained.
#
# To get a better understanding of the two steps, we’ll explore them separately. Note that we will use some methods
# that perform parts of the tokenization pipeline separately to show you the intermediate results of those steps,
# but in practice, you should call the tokenizer directly on your inputs

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate when lessons are not updated!",
]
# multiple input sequences passed to the tokenizer must result in tensors of the same length, therefore, padding the
# input sequences is required.
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(inputs)

# Even the tokenizer step has substeps, the following code blocks are the substeps of the above tokenizer code step
# The tokenization process is done by the tokenize() method of the tokenizer:
# This tokenizer is a subword tokenizer: it splits the words until it obtains tokens that can be represented by its vocabulary.
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."
tokens = tokenizer.tokenize(sequence, return_tensors='pt')

print(tokens)

# The conversion to input IDs is handled by the convert_tokens_to_ids() tokenizer method:
ids = tokenizer.convert_tokens_to_ids(tokens)

print(ids)

# Decoding is going the other way around: from vocabulary indices, we want to get a string. This can be done with
# the decode() method as follows:
# Note that the decode method not only converts the indices back to tokens, but also groups together the tokens that
# were part of the same words to produce a readable sentence.
decoded_string = tokenizer.decode(ids)
print(decoded_string)

# Now to pass the tokenized number ids inputs into the model, the id numbers need to be converted to appropriate tensor
# framework
# Transformers models expect multiple sentences / input sequences by default. Here we tried to do everything the tokenizer did behind the
# scenes when we applied it to a sequence. In the tokenizer pipeline, the tokenizer didn’t just convert
# the list of input IDs into a tensor, it added a dimension on top of it so we'll do the same in this step
model_inputs = torch.tensor([ids])  # use ([ids]) instead of (ids)
print("Model input ids:", model_inputs)

# model_output = model(model_inputs)  doesn't work for some reason even though it's directly from the HF LLM course

# print("Logits:", model_output.logits)


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

# OK, we all done with the tokenizer steps and pipeline, now back to figuring out how to sentiment analysis of our
# input sentence sequences
# we will need a model with a sequence classification head (to be able to classify the sentences as positive or negative).

outputs = model(**inputs) # we're now using the model inputs created from the previous tokenizer pipeline

# Since we have just two sentences and two labels, the result we get from our model is of shape 2 x 2.
print(outputs.logits.shape)

# post processing the model output
print(outputs.logits)

# Our model predicted [-1.5607, 1.6123] for the first sentence and [ 4.1692, -3.3464] for the second one. Those are not
# probabilities but logits, the raw, unnormalized scores outputted by the last layer of the model. To be converted to
# probabilities, they need to go through a SoftMax layer (all 🤗 Transformers models output the logits, as the loss
# function for training will generally fuse the last activation function, such as SoftMax, with the actual loss function,
# such as cross entropy):

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

