import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('gutenberg')

import torch
from transformers import pipeline


book = nltk.corpus.gutenberg.raw('carroll-alice.txt')
chapters = book.split('CHAPTER')
summaries = ""

# Pretrained LongT5 transformer model trained with BOOKSUM
summarizer = pipeline(
    "summarization",
    "pszemraj/long-t5-tglobal-base-16384-book-summary",
    device=0 if torch.cuda.is_available() else -1,
)
# Number of chapter to summarize
numChapters = len(chapters)

# Summarize each chapter of the book using LongT5 model
for i in range(1, numChapters):
    temp = chapters[i]
    title = temp.split('\n')[0]

    chapSummary = summarizer(chapters[i])
    chapSummary = chapSummary[0]["summary_text"]
    chapSummary = chapSummary.split('. ')
    summaries += f'Chapter{title}\n'

    for l in range(len(chapSummary)):
        summaries += f'    {chapSummary[l]}\n'
    summaries += "-------------------------------------\n"


with open('chapterSummaries.txt', 'w') as f:
    f.write(summaries)


