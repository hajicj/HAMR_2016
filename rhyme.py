#!/usr/bin/env python
"""This is a script that tests rhymes.

Input:

  rhyme.py -w word1 word2

Outputs:

  rhyming data

"""
from __future__ import print_function, unicode_literals
import argparse
import collections
import logging
import pprint
import sys
import time

import nltk

__version__ = "0.0.1"
__author__ = "Jan Hajic jr."


##############################################################################

# We need a pronunciation substitution table.


##############################################################################


def rhyme(w, level):
    entries = nltk.corpus.cmudict.entries()
    syllables = [(word, phoneme) for word, phoneme in entries if word == w]
    rhymes = []
    for (word, syllable) in syllables:
        rhymes += [word for word, pron in entries if pron[-level:] == syllable[-level:]]
    return set(rhymes)


def do_they_rhyme(word1, word2):
    # first, we don't want to report 'glue' and 'unglue' as rhyming words
    # those kind of rhymes are LAME
    if word1.find(word2) == len(word1) - len(word2):
        return False
    if word2.find(word1) == len(word2) - len(word1):
        return False

    return word1 in rhyme(word2, 1)


def pron_bleu(word1, word2, prondict):
    """Returns the maximum BLEU score of two words' pronunciations.
    Supply the prondict as a dictionary with word keys and list of prons
    as values."""
    prons_1 = prondict[word1]
    prons_2 = prondict[word2]

    logging.info('Prons 1: {0}'.format(prons_1))
    logging.info('Prons 2: {0}'.format(prons_2))

    best_score = -1.0
    best_p1 = None
    best_p2 = None
    for p1 in prons_1:
        for p2 in prons_2:
            bleu = nltk.translate.bleu_score.sentence_bleu([p2], p1)
            if bleu > best_score:
                best_p1 = p1
                best_p2 = p2
                best_score = bleu

    return best_p1, best_p2, best_score


def word_rhyming_table(words, prondict=None, pair_score_fn=pron_bleu):

    if prondict is None:
        prondict = collections.defaultdict(list)
        for word, syl in nltk.corpus.cmudict.entries():
            prondict[word].append(syl)

    words = args.words

    # Your code goes here
    pair_scores = {}
    pair_prons = {}
    for w1 in words:

        if w1 not in prondict:
            raise ValueError('Word {0} not found in the dictionary.'.format(w1))
        for w2 in words:
            if w2 not in prondict:
                raise ValueError('Word {0} not found in the dictionary.'.format(w2))

            if w1 == w2:
                continue
            if ((w1, w2) in pair_scores) or ((w2, w1) in pair_scores):
                continue

            p1, p2, score = pair_score_fn(w1, w2, prondict)
            pair_scores[(w1, w2)] = score
            pair_prons[(w1, w2)] = (p1, p2)

    return pair_scores, pair_prons


##############################################################################


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__, add_help=True,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-w', '--words', action='store', nargs='+',
                        help='space-separated list of words.')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Turn on INFO messages.')
    parser.add_argument('--debug', action='store_true',
                        help='Turn on DEBUG messages.')

    return parser


def main(args):
    logging.info('Starting main...')
    _start_time = time.clock()

    cmudict = collections.defaultdict(list)
    for word, syl in nltk.corpus.cmudict.entries():
        cmudict[word].append(syl)

    words = args.words

    # Your code goes here
    pair_scores, pair_prons = word_rhyming_table(words, prondict=cmudict, pair_score_fn=pron_bleu)

    for w1, w2 in pair_scores:
        s = pair_scores[(w1, w2)]
        p1, p2 = pair_prons[(w1, w2)]
        print('{0} :: {1} ::\t{2}\t::\t{3} :: {4}'.format(w1, w2, s, p1, p2))

    _end_time = time.clock()
    logging.info('[XXXX] done in {0:.3f} s'.format(_end_time - _start_time))


##############################################################################


if __name__ == '__main__':
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    if args.debug:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

    main(args)
