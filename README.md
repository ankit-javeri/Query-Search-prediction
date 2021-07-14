# Query-Search-prediction - NLP
Display a dropdown of suggestions from database for searched queries.

# Technique Used
The search queries available in database are converted into vectors by finding TFIDF for each document and each term. Then cosine similarity between search query and all documents is computed and the top most similar documents are displayed as suggestions.

## Here's an example image of expected output
![alt text](https://www.google.com/url?sa=i&url=https%3A%2F%2Fsearchengineland.com%2Fhow-google-instant-autocomplete-suggestions-work-62592&psig=AOvVaw1UuPl-3EDsC3MUe06om6oy&ust=1626353414842000&source=images&cd=vfe&ved=0CAgQjRxqFwoTCJinlt_M4vECFQAAAAAdAAAAABAJ)

