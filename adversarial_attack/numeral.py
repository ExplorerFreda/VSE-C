""" only replace head nouns """
from functools import reduce
from os.path import join as pjoin
from pattern import *
import spacy
from word2number import w2n
import argparse
import os

# load spacy
nlp = spacy.load('en')


def detect_number(text):
    left = 0
    detect_results = list()
    while left < len(text):
        right = left + 1
        if text[left].pos_ == 'NUM':
            right = left + 1
            while right < len(text) and text[right].pos_ == 'NUM':
                right += 1
            if right == len(text):
                continue
            number = ' '.join([word.text for word in text[left: right]])
            if not valid(number):
                left = right
                continue
            num_range = [left, right]
            while right < len(text) and text[right].pos_ in ['ADJ', 'CCONJ']:
                right += 1
            if right == len(text):
                continue
            if text[right].pos_ == 'NOUN':
                while right < len(text) and text[right].pos_ == 'NOUN':
                    right += 1
                n = text[right - 1].text
                n_index = right - 1
            else:
                left = right
                continue
            detect_results.append([number, num_range, n, n_index])
        elif text[left].pos_ == 'DET' and (text[left].text == 'an' or text[left].text == 'a'):
            right = left + 1
            num_range = [left, right]
            while right < len(text) and text[right].pos_ in ['ADJ', 'CCONJ']:
                right += 1
            if right == len(text):
                continue
            if text[right].pos_ == 'NOUN':
                while right < len(text) and text[right].pos_ == 'NOUN':
                    right += 1
                n = text[right - 1].text
                n_index = right - 1
            else:
                left = right
                continue
            detect_results.append([text[left].text, num_range, n, n_index])
        left = right
    return detect_results


def valid(word):
    # check number
    try:
        numbers = [w2n.word_to_num(it) for it in word.split()]
        if len(numbers) > 1:
            return False
    except ValueError:
        return False
    # check English
    chars = map(lambda c: 'a' <= c <= 'z', word)
    return reduce(lambda a, b: a or b, chars, False)


prefix_list = ['test', 'dev', 'train']


# rule:
#   1. NUM string
#   2. after NUM should be NOUN
#   3. Noun described by NUM should be the last one of the first NOUN string after NUM
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default='resnet152_numex_precomp')
    parser.add_argument('--data_path', type=str, default='../data/')
    args = parser.parse_args()
    # load all coco captions
    coco_captions = list()
    number_set = set()
    for prefix in prefix_list:
        os.system('mkdir {:s}/{:s}/{:s}_ex'.format(args.data_path, args.data_name, prefix))
        path_coco_captions = pjoin('{:s}/{:s}/'.format(args.data_path, args.data_name), prefix + '_caps.txt')
        coco_captions += open(path_coco_captions).readlines()
    # count word-frequency for nouns and adjectives
    frequency = dict()
    number_set = {'two', 'six', 'eight', 'five', 'fifteen',
                  'nine', 'a', 'an', 'four', 'three',
                  'seven', 'fourteen', 'twenty', 'eighteen', 'ten',
                  'one', 'thirty', 'eleven', 'twelve', 'thirteen',
                  'sixteen', 'seventeen', 'nineteen', 'sixty', 'forty',
                  'fifty', 'seventy', 'eighty'}
    # process_numbers
    number_dict = dict()
    for item in number_set:
        if item in ['a', 'an', 'one']:
            number_dict[item] = 1
        else:
            try:
                number_dict[item] = w2n.word_to_num(item)
            except ValueError:
                print('value error:', item)
    # replace these words with words which are similar to themselves but have different meanings
    for prefix in prefix_list:
        path_coco_captions = pjoin('{:s}/{:s}/'.format(args.data_path, args.data_name), prefix + '_caps.txt')
        real_captions = open(path_coco_captions).readlines()
        for i, caption in enumerate(real_captions):
            replacement_cnt = 0
            fout = open('{:s}/{:s}/{:s}_ex/{:d}.txt'.format(args.data_path, args.data_name, prefix, i), 'w')
            text = nlp(caption.lower())
            number_info = detect_number(text)
            # replace
            for item in number_info:
                number_range = item[1]
                w = item[0]
                noun = item[2]
                noun_index = item[3]
                if w not in number_dict:
                    continue
                # enumerate all possible words for replacement
                for r in number_dict:
                    if number_dict[r] == number_dict[w]:
                        continue
                    word_list = [x.text for x in text]
                    # singularize or pluralize
                    if number_dict[r] == 1:
                        new_noun = singularize(noun)
                    elif number_dict[w] == 1:
                        new_noun = pluralize(noun)
                    else:
                        new_noun = noun
                    word_list[noun_index] = new_noun
                    new_word_list = word_list[:number_range[0]] + [r] + word_list[number_range[1]:]
                    replacement = ' '.join(new_word_list)
                    fout.write(replacement)
                    replacement_cnt += 1
            if prefix in ['train', 'dev']:
                for j in range(replacement_cnt, 5):
                    fout.write(' '.join(['<unk>' for _ in caption.strip().split()]) + '\n')
            fout.close()
            print(prefix, i, replacement_cnt)
