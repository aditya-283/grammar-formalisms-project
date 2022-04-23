#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Set up
# pip install stanza (or conda install -c stanfordnlp stanza)
# and download the English model by import stanza; stanza.download('en')

# Version info
# stanza-1.3.0
# model: en default

# How to use
# In this directory:
# python parse_recipes.py

from os import listdir
from os import path
import argparse
import json
import logging

from stanza.utils.conll import CoNLL
from tqdm import tqdm
import stanza

verbose = False
logger = None


def init_logger(name='logger'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    log_fmt = '%(asctime)s/%(name)s[%(levelname)s]: %(message)s'
    logging.basicConfig(format=log_fmt)
    return logger


def main(args):
    global verbose
    verbose = args.verbose

    # The input sentences are already tokenized (tokenize_pretokenized=True, tokenize_no_ssplit=True)
    nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma,depparse',
                          tokenize_pretokenized=True, tokenize_no_ssplit=True)

    if verbose:
        logger.info(f'In: {args.path_input}')
        logger.info(f'Out: {args.path_output}')

    of = open(args.path_output, 'w')
    with open(args.path_input) as f:
        for sid, line in enumerate(f):
            text = line.strip()
            if len(text) == 0:
                continue
            # https://stanfordnlp.github.io/stanza/tokenize.html#start-with-pretokenized-text
            tokens = [tok.strip() for tok in text.split() if tok.strip()]
            doc = nlp([tokens])

            # Convert into the CoNLL-U format and output
            of.write(f'# sentid = {sid}\n')
            of.write(f'# text = {text}\n')
            of.write('\n'.join('\t'.join(row) for row in CoNLL.convert_dict(doc.to_dict())[0]) + '\n')
            of.write('\n')

    of.close()
    return 0


if __name__ == '__main__':
    logger = init_logger('Parse')
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', dest='path_input',
                        default='recipe_sentences_2022-04-12.txt', help='path to an input file')
    parser.add_argument('-o', '--output', dest='path_output',
                        default='recipe_sentences_2022-04-12.conllu',
                        help='path to an output file')
    parser.add_argument('-v', '--verbose',
                        action='store_true', default=False,
                        help='verbose output')
    args = parser.parse_args()
    main(args)
