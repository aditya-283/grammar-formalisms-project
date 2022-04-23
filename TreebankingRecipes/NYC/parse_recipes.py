#!/usr/bin/env python
# -*- coding: utf-8 -*-

from nltk.tokenize.treebank import TreebankWordDetokenizer
from os import listdir
from os import path
from stanza.utils.conll import CoNLL
from tqdm import tqdm
import argparse
import json
import logging
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

    nlp = stanza.Pipeline(lang='en', tokenize_pretokenized=True, tokenize_no_ssplit=True)
    detokenizer = TreebankWordDetokenizer()

    of = open(args.path_output, 'w')
    for filename in tqdm(listdir(args.dir_input)):
        if not filename.endswith('.json'):
            continue
        with open(path.join(args.dir_input, filename)) as f:
            dat = json.load(f)
        docid = dat['id']
        n_sents = max(map(int, dat['text'].keys()))
        for sid in range(n_sents):
            tokens = [tok.strip() for tok in dat['text'][str(sid)] if tok.strip()]
            text = detokenizer.detokenize(tokens, convert_parentheses=True)
            doc = nlp([tokens])
            of.write(f'# docid = {docid}\n')
            of.write(f'# sentid = {sid}\n')
            of.write(f'# text = {text}\n')
            if len(doc.sentences) > 1:
                import pdb
                pdb.set_trace()
            of.write('\n'.join('\t'.join(row) for row in CoNLL.convert_dict(doc.to_dict())[0]) + '\n')
            of.write('\n')

    of.close()
    return 0


if __name__ == '__main__':
    logger = init_logger('Parse')
    parser = argparse.ArgumentParser()
    parser.add_argument('dir_input', help='path to input file')
    parser.add_argument('-o', '--output', dest='path_output',
                        help='path to output file')
    parser.add_argument('-v', '--verbose',
                        action='store_true', default=False,
                        help='verbose output')
    args = parser.parse_args()
    main(args)
