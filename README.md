# Query-Search-prediction - NLP
Display a dropdown of suggestions from database for searched queries.

# Technique Used
The search queries available in database are converted into vectors by finding TFIDF for each document and each term. Then cosine similarity between search query and all documents is computed and the top most similar documents are displayed as suggestions.

## Here's an example image of expected output
![auto_suggest](https://user-images.githubusercontent.com/77067492/125628322-3da7001d-5f83-40b7-b84f-8e2cf4011282.jpeg)

[Image Source](https://www.google.com/url?sa=i&url=https%3A%2F%2Fmedium.com%2Frelated-works-inc%2Fbootstrapping-autosuggest-c1ca3edaf1eb&psig=AOvVaw0djNL-g_CqU1lDgMTIgMum&ust=1626354829675000&source=images&cd=vfe&ved=2ahUKEwjb2b2A0uLxAhWZv2MGHbe7DvkQjRx6BAgAEAw)
