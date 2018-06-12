""" only replace head nouns """
from os.path import join as pjoin
import spacy
import argparse
import os

# load spacy
nlp = spacy.load('en')


def match(orig_text, pattern_text):
    for i_ in range(len(orig_text)):
        if orig_text[i_] != pattern_text[i_]:
            return False
    return True


def English(word):
    for k_ in range(len(word)):
        if word[k_] != ' ' and not ('a' <= word[k_] <= 'z'):
            return False
    return True


def detect_noun_chunks(text_info):
    noun_chunks_ = list(text_info.noun_chunks)
    r_ = [[False for _ in range(len(noun_chunks_))] for __ in range(len(noun_chunks_))]
    for j_ in range(len(noun_chunks_)):
        for k_ in range(j_+1, len(noun_chunks_)):
            if noun_chunks_[j_].end + 1 < noun_chunks_[k_].start:
                continue
            if (noun_chunks_[j_].end == noun_chunks_[k_].start) or (
                            text_info[noun_chunks_[j_].end].text in [',  ',  'with',  'and',  'or']):
                r_[j_][k_] = True
                r_[k_][j_] = True
    # compute transitive closure
    for p_ in range(len(noun_chunks_)):
        for j_ in range(len(noun_chunks_)):
            for k_ in range(len(noun_chunks_)):
                if r_[j_][p_] and r_[p_][k_]:
                    r_[j_][k_] = True
    return noun_chunks_, r_


prefix_list = ['dev', 'test', 'train']


# rule: no swap for noun chunks connected by comma, with, and, or
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default='resnet152_relationex_precomp')
    parser.add_argument('--data_path', type=str, default='../data/')
    args = parser.parse_args()
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
        {'toronto'}]

    prep_replace_dict = dict()
    for subset in prep_set:
        for word in subset:
            if word in prep_replace_dict:
                continue
            prep_replace_dict[word] = set()
            for sub in prep_set:
                if word in sub:
                    continue
                for w in sub:
                    prep_replace_dict[word].add(w)
    for prefix in prefix_list:
        os.system('mkdir {:s}/{:s}/{:s}_ex'.format(args.data_path, args.data_name, prefix))
        path_coco_captions = pjoin('{:s}/{:s}/'.format(args.data_path, args.data_name), prefix + '_caps.txt')
        real_captions = open(path_coco_captions).readlines()
        for i, caption in enumerate(real_captions):
            replacement_cnt = 0
            fout = open('{:s}/{:s}/{:s}_ex/{:d}.txt'.format(args.data_path, args.data_name, prefix, i), 'w')
            text = nlp(caption.lower())
            noun_chunks, no_swap = detect_noun_chunks(text)
            for j in range(len(noun_chunks)):
                for k in range(j + 1, len(noun_chunks)):
                    if no_swap[j][k]:
                        continue
                    new_word_list = \
                        list(text[:noun_chunks[j].start]) + \
                        list(noun_chunks[k]) + \
                        list(text[noun_chunks[j].end: noun_chunks[k].start]) + \
                        list(noun_chunks[j]) + \
                        list(text[noun_chunks[k].end:])
                    new_caption = ' '.join([word.text for word in new_word_list])
                    replacement_cnt += 1
                    fout.write(new_caption)

            new_word_list = [item.text for item in text]
            for j, word in enumerate(text):
                if word.pos_ == 'ADP' and (word.text in prep_replace_dict):
                    for replace_word in prep_replace_dict[word.text]:
                        new_word_list[j] = replace_word
                        fout.write(' '.join(new_word_list))
                        new_word_list[j] = word.text
                        replacement_cnt += 1
            if prefix in ['train', 'dev']:
                for j in range(replacement_cnt, 5):
                    fout.write(' '.join(['<unk>' for _ in caption.strip().split()]) + '\n')
            fout.close()
            print(prefix, i, replacement_cnt)
