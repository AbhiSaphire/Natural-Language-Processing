from goldman_emma_raw import goldman_docs
from henson_matthew_raw import henson_docs
from wu_tingfang_raw import wu_docs

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

friends_docs = goldman_docs + henson_docs + wu_docs
friends_labels = [1] * 154 + [2] * 141 + [3] * 166

print(goldman_docs[100])
print(hensorn_docs[50])


mystery_postcard = """
My friend,
From the 10th of July to the 13th, a fierce storm raged, clouds of
freeing spray broke over the ship, incasing her in a coat of icy mail,
and the tempest forced all of the ice out of the lower end of the
channel and beyond as far as the eye could see, but the _Roosevelt_
still remained surrounded by ice.
Hope to see you soon.
"""

bow_vectorizer = CountVectorizer()

friends_vectors = bow_vectorizer.fit_transform(friends_docs)

mystery_vector = bow_vectorizer.transform([mystery_postcard])

friends_classifier = MultinomialNB()

friends_classifier.fit(friends_vectors, friends_labels)

probabilistic_predictions = friends_classifier.predict_proba(mystery_vector)
print(probabilistic_predictions)

predictions = friends_classifier.predict(mystery_vector)
mystery_friend = predictions[0] if predictions[0] else "someone else"

print("The postcard was from {}!".format(mystery_friend))