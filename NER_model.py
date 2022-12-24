import nltk
from nltk.tokenize import sent_tokenize
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('gutenberg')

from tqdm import tqdm
import string
from collections import Counter
from flair.models import SequenceTagger
from flair.data import Sentence

book = nltk.corpus.gutenberg.raw('carroll-alice.txt')
chapters = book.split('CHAPTER')

summary = ''
print(f"NER MODEL")
# For each chapter, run NER
for i in range(1, len(chapters)):

    temp = chapters[i]
    title = temp.split('\n')[0]
    chapters[i] = chapters[i].replace('\n', ' ')
    chapters[i] = chapters[i].replace('\r', ' ')
    chapters[i] = chapters[i].replace('\'', ' ')
    sent = sent_tokenize(chapters[i])

    # Flair named entity recognition model
    tagger = SequenceTagger.load('ner')

    # Get all the characters names and locations
    characters = []
    locations = []
    for line in tqdm(sent):
        sentence = Sentence(line)

        tagger.predict(sentence)
        for entity in sentence.get_spans('ner'):

            # If person, add to characters list
            if entity.get_label("ner").value == 'PER':
                characters.append(entity.text)

            # If location, add to location list
            elif entity.get_label("ner").value == 'LOC':
                locations.append(entity.text)


    # Remove any punctuation within the names
    names = []
    for name in characters:
        names.append(name.translate(str.maketrans('', '', string.punctuation)))

    # List characters by the frequency with which they are mentioned
    result = [item for items, c in Counter(characters).most_common() for item in [items] * c]


    common = []
    main_freq = []

    # Manually remove words that are not character names from our list
    not_names = ['Well', 'Ive', 'Five', 'Theyre', 'Dont', 'Wow', 'Ill', 'Miss', 'Hush', 'Yes', ]

    for n, c in Counter(names).most_common():
        if  n not in not_names:
            main_freq.append((n, c))
            common.append(n)


    summary += f"Chapter{title}:\n    Character List: {common}\n    Locations: {list(set(locations))}\n"
    summary += "---------------------------------------------\n"

with open('charactersLocations.txt', 'w') as f:
    f.write(summary)

