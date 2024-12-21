import re
from collections import Counter

import requests

# GitHub url that holds the desired dataset.
PTB_TRAIN = "https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.train.txt"


def get_ptb_train() -> Counter:
    # Use requests to obtain the content.
    response = requests.get(PTB_TRAIN)
    response.raise_for_status()

    # Find all the words from the requested context.
    words = re.findall(r"\b[\w'-]+\b", response.text.lower())

    # Return the word: appearance count.
    return Counter(words)
