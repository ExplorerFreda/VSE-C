""" only replace head nouns """
from nltk.corpus import wordnet as wn
from os.path import join as pjoin
import spacy
import argparse
import os
import random


# load spacy
nlp = spacy.load('en')


def valid(word, replacement):
    if replacement.lower() != replacement:
        return False
    synset_word = wn.synsets(word)
    synset_replacement = wn.synsets(replacement)
    for item_1 in synset_word:
        for item_2 in synset_replacement:
            if item_1 == item_2:
                return False
    # one-step hypernymy/hyponymy check
    for item_1 in synset_word:
        for subitem in item_1.hypernyms():
            for item_2 in synset_replacement:
                if item_2 == subitem:
                    return False
        for subitem in item_1.hyponyms():
            for item_2 in synset_replacement:
                if item_2 == subitem:
                    return False
    return True


def English(word):
    for k in range(len(word)):
        if word[k] != ' ' and not ('a' <= word[k] <= 'z'):
            return False
    return True


def tag_noun_chunk(text):
    text = text
    text_info = nlp(text)
    ws = list([x.text for x in text_info])
    replace_tag = [False] * len(ws)
    for chunk in text_info.noun_chunks:
        if not English(chunk.text):
            continue
        replace_tag[chunk.end - 1] = True
    return ws, replace_tag


prefix_list = ['train', 'dev', 'test']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default='resnet152_nounex_precomp')
    parser.add_argument('--data_path', type=str, default='../data')
    args = parser.parse_args()
    # load all coco captions
    coco_captions = list()
    for prefix in prefix_list:
        os.system('mkdir {:s}/{:s}/{:s}_ex'.format(args.data_path, args.data_name, prefix))
        path_coco_captions = pjoin('{:s}/{:s}/'.format(args.data_path, args.data_name), prefix + '_caps.txt')
        coco_captions += open(path_coco_captions).readlines()
    # count word-frequency for nouns and adjectives
    frequency = dict()
    for i in range(0, len(coco_captions), 10000):
        # default: treat the last word of a noun chunk as its head
        caption = ' '.join(coco_captions[i: min(i+10000, len(coco_captions))])
        caption = nlp(caption.lower())
        for word in caption:
            if word.pos_ == 'NOUN' and word.lemma_ != '--PRON--':
                word = word.text
                if word in frequency:
                    frequency[word] += 1
                else:
                    frequency[word] = 1
        print(i)
    # sort and select most frequent nouns
    frequency_pair = sorted([(token, frequency[token]) for token in frequency], key=lambda x: x[1], reverse=True)
    frequency_pair = list(filter(lambda x: x[1] >= 200, frequency_pair))  # frequency >= 200
    # load concreteness
    concreteness = list(filter(
        lambda x: float(x[1]) > 0.6,
        [line.split() for line in open('{:s}/concreteness.txt'.format(args.data_path)).readlines()]))
    concrete_words = set([item[0] for item in concreteness])
    # filter out concrete words
    frequency_pair = list(filter(lambda x: x[0] in concrete_words, frequency_pair))
    frequent_words = [item[0] for item in frequency_pair]
    frequent_words_set = set(frequent_words)
    # get candidates for replacement
    candidate_words = dict()
    for w in frequent_words:
        candidate_words[w] = list(filter(
            lambda x: valid(w, x),
            [item for item in frequent_words_set]))
    # replace these words with words which are similar to themselves but have different meanings
    for prefix in prefix_list:
        path_coco_captions = pjoin('{:s}/{:s}/'.format(args.data_path, args.data_name), prefix + '_caps.txt')
        real_captions = open(path_coco_captions).readlines()
        invalid_word_set = set()
        for i in range(len(real_captions)):
            caption = real_captions[i]
            # load 5 recent captions for invalid word set
            if i % 5 == 0:
                invalid_word_set = set()
                for j in range(i, i + 5):
                    for w in real_captions[j].split():
                        invalid_word_set.add(w)
            replacement_cnt = 0
            fout = open('{:s}/{:s}/{:s}_ex/{:d}.txt'.format(args.data_path, args.data_name, prefix, i), 'w')
            caption = caption.lower()
            words, noun_chunk_tag = tag_noun_chunk(caption)
            replaced_words = [w for w in words]
            all_replacements = list()
            for j, token in enumerate(words):
                if not noun_chunk_tag[j]:
                    continue
                if token in candidate_words:
                    for candidate in candidate_words[token]:
                        if candidate in invalid_word_set:  # skip invalid words
                            continue
                        replaced_words[j] = candidate
                        all_replacements.append(' '.join(replaced_words))
                        replacement_cnt += 1
                replaced_words[j] = words[j]
            random.shuffle(all_replacements)
            for line in all_replacements:
                fout.write(line)
            if prefix in ['train', 'dev']:
                for j in range(replacement_cnt, 5):
                    fout.write(' '.join(['<unk>' for _ in caption.strip().split()]) + '\n')
            fout.close()
            print(prefix, i, replacement_cnt)
