""" Topics are a way to categorize and organize content. You can use topics to help users find relevant content,
recommend related content, and understand the relationships between different pieces of content. For example,
you can find similar lessons based on their topics.

There are many ways to extract topics from unstructured text. You could use an LLM and ask it to summarize
the topics from the text. A more straightforward approach is to identify all the nouns in the text and
use them as topics.

The Python NLP (natural language processing) library, textblob, can extract noun phrases from text. """


from textblob import TextBlob

phrase = "You can extract topics from phrases using TextBlob"

topics = TextBlob(phrase).noun_phrases

print(topics)