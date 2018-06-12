import json
import argparse

import spacy
from multiprocessing import Pool

nlp = spacy.load('en')

parser = argparse.ArgumentParser()
parser.add_argument('--input', required=True)
parser.add_argument('--output', required=True)
args = parser.parse_args()


def isalpha(word):
    for k in range(len(word)):
        if word[k] != ' ' and not ('a' <= word[k] <= 'z'):
            return False
    return True


prep_set = [
        {'towards', 'toward', 'beyond', 'to'},
        {'behind', 'after',  'past'},
        {'outside', 'out'},
        {'on', 'upon', 'up', 'un', 'atop', 'onto', 'over', 'above', 'beyond'},
        {'in', 'within', 'among', 'at', 'during', 'into', 'inside', 'from', 'among', 'between'},
        {'if'},
        {'with', 'by', 'beside'},
        {'around', 'like'},
        {'underneath', 'under', 'beneath', 'down', 'below'},
        {'to', 'for', 'of'},
        {'about', 'within'},
        {'because', 'as', 'for'},
        {'as', 'like'},
        {'near', 'next', 'beside'},
        {'while'},
        {'though'},
        {'thru', 'through'},
        {'besides'},
        {'against', 'next', 'to'},
        {'along', 'during', 'across', 'while'},
        {'ny'},
        {'off',  'out'},
        {'sleeve'},
        {'without'},
        {'than'},
        {'photographer'},
        {'before'},
        {'across'},
        {'toronto'}
]

prep_set = {p for l in prep_set for p in l}


def process(caption):
    caption = str(caption.strip())
    text_info = nlp(caption)
    ws = list([x.text for x in text_info])
    replace_noun = [False] * len(ws)
    for chunk in text_info.noun_chunks:
        if not isalpha(chunk.text):
            continue
        replace_noun[chunk.end - 1] = True
   
    replace_prep = [False] * len(ws)
    if len(text_info) != len(caption.split()):
        print(list(enumerate(text_info)), caption)
    for i, word in enumerate(text_info):
        if word.pos_ == 'ADP' and word.text in prep_set:
            replace_prep[i] = True

    replace_noun = [i for i, v in enumerate(replace_noun) if v]
    replace_prep = [i for i, v in enumerate(replace_prep) if v]
    replace_all = replace_noun + replace_prep

    return dict(sentence=' '.join(ws), replace=replace_all, replace_noun=replace_noun, replace_prep=replace_prep)


def main():
    with open(args.input) as f:
        lines = f.readlines()
    results = Pool().map(process, lines)
    with open(args.output, 'w') as f:
        json.dump(results, f)
    from IPython import embed; embed()


if __name__ == '__main__':
    main()

