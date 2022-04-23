#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Setup
# pip install conllu

from collections import defaultdict
import argparse
import logging

import conllu

verbose = False
logger = None


def init_logger(name='logger'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    log_fmt = '%(asctime)s/%(name)s[%(levelname)s]: %(message)s'
    logging.basicConfig(format=log_fmt)
    return logger


def get_pos_pattern(tokens):
    FOCAL_UPOS = {'ADJ', 'ADP', 'ADV', 'AUX', 'NOUN', 'PROPN', 'VERB'}
    buff = []
    for token in tokens:
        if token['upos'] not in FOCAL_UPOS:
            continue
        if len(buff) > 0 and buff[-1] == token['upos']:
            continue
        buff.append(token['upos'])
    return ' '.join(buff)

def main(args):
    global verbose
    verbose = args.verbose

    pos2sentences = defaultdict(list)
    sentence_counter = defaultdict(int)
    if verbose:
        logger.info(f'In: {args.path_input}')
        logger.info(f'Out: {args.path_output}')

    counter = 0
    with open(args.path_input) as f:
        for sentence in conllu.parse_incr(f):
            pos_pat = get_pos_pattern(sentence)
            pos2sentences[pos_pat].append(sentence)
            counter += 1
    if verbose:
        logger.info(f'Read {counter} sentences ({len(pos2sentences)} UPOS patterns)')

    of = open(args.path_output, 'w')
    new_sentid = 1
    while sum([len(sents) for _, sents in pos2sentences.items()]) > 0:
        for pat, sents in sorted(pos2sentences.items(),
                                 key=lambda t: (-len(t[1]), len(t[0].split()))):
            if len(sents) == 0:
                continue
            sent = sents.pop(0)
            sent.metadata['original_sentid'] = sent.metadata['sentid']
            sent.metadata['sentid'] = str(new_sentid)
            new_sentid += 1
            of.write(sent.serialize() + '\n')
    of.close()

    return 0


if __name__ == '__main__':
    logger = init_logger('Reorder')
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', dest='path_input', default='recipe_sentences_2022-04-12.conllu', help='path to an input file')
    parser.add_argument('-o', '--output', dest='path_output', default='recipe_sentences_2022-04-12.reordered-v1.conllu', help='path to an output file')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose output')
    args = parser.parse_args()
    main(args)
