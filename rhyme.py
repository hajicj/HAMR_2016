#!/usr/bin/env python
"""This is a script that tests rhyme scoring functions.

Input:

  rhyme.py -t take a step back hey are you really gonna hack a full stack in a day while on crack

Outputs:

  rhyming data:
    - count
    - locations (density?)
    - distances

Parameters:

    - strip stress

Relevant ideas
--------------

Novelty of vocabulary.
Not repeating yourself.
Rhymes should be varied enough, but more consistent lines of rhyming are also appreciated.

"""
from __future__ import print_function, unicode_literals
import argparse
import collections
import logging
import pprint
import string
import sys
import time

import itertools
import matplotlib.pyplot as plt
import nltk
import numpy

from pronounce import Pronounce

__version__ = "0.0.1"
__author__ = "Jan Hajic jr."


##############################################################################


class ARPAbet(object):
    """Plain Old Data class for the ARPAbet phonemes."""
    phonemes = {
        'AA',
        'AE',
        'AH',
        'AO',
        'AW',
        'AY',
        'B',
        'CH',
        'D',
        'DH',
        'EH',
        'ER',
        'EY',
        'F',
        'G',
        'HH',
        'IH',
        'IY',
        'JH',
        'K',
        'L',
        'M',
        'N',
        'NG',
        'OW',
        'OY',
        'P',
        'R',
        'S',
        'SH',
        'T',
        'TH',
        'UH',
        'UW',
        'V',
        'W',
        'Y',
        'Z',
        'ZH',
    }

    vowels = {
        'AA',
        'AE',
        'AH',
        'AO',
        'AW',
        'AY',
        'EH',
        'ER',
        'EY',
        'IH',
        'IY',
        'JH',
        'OW',
        'OY',
        'UH',
        'UW',
        'Y',
    }

    @staticmethod
    def is_vowel(p):
        if strip_stress([p])[0] in ARPAbet.vowels:
            return True
        else:
            return False

    @staticmethod
    def is_consonant(p):
        return not ARPAbet.is_vowel(p)


##############################################################################


def tokenize(text):
    tokens = nltk.tokenize.wordpunct_tokenize(text.translate(string.punctuation).lower())
    return tokens


def get_prons(words, prondict, unknown_pron_client=None):
    """Collect pronunciations for each word (as a list of ARPAbet lists).
    For unknown words, queries the Logios server at CMU.
    """
    prons = []
    for w in words:
        if w not in prondict:
            if unknown_pron_client is None:
                unknown_pron_client = Pronounce()
            unknown_pron_client.add(w)
            ps = unknown_pron_client.p(add_fake_stress=True)
        else:
            ps = prondict[w]
        prons.append(ps)
    return prons


def strip_stress(pron):
    """Removes the stress markers from the phones.

    >>> pron = ['AA1', 'R', 'G', 'Y', 'UH0']
    >>> strip_stress(pron)
    [u'AA', u'R', u'G', u'Y', u'UH']
    """
    out = []
    for p in pron:
        if p[-1] in '0123':
            out.append(p[:-1])
        else:
            out.append(p)
    return out


def strip_consonants(pron):
    return [p for p in pron if ARPAbet.is_vowel(p)]


def disambiguate_indefinite_article_prons(words, prons):
    """The indefinite article has two pronunciations, but we can deterministically say
    which one will be used based on the first phoneme of the next word.
    Works on the entire words/prons sequences.
    """
    INDEFINITE_ARTICLE_PRONS = [['AH'], ['EY']]
    INDEFINITE_ARTICLE_LONG_PRON = [['EY']]
    INDEFINITE_ARTICLE_SHORT_PRON = [['AH']]

    out_words, out_prons = [], []
    for i, (w, ps) in enumerate(itertools.izip(words[:-1], prons[:-1])):
        if w.lower() == 'a':
            next_ps = prons[i + 1]
            # logging.debug('Encountered indefinite article, next prons: {0}'.format(next_ps))
            is_long = False
            for p in next_ps:
                # logging.debug('Testing pron: first phoneme {0}'.format(p[0]))
                if ARPAbet.is_vowel(p[0]):
                    logging.debug('\tIs vowel!')
                    is_long = True

            if is_long:
                out_pron = INDEFINITE_ARTICLE_LONG_PRON
            else:
                out_pron = INDEFINITE_ARTICLE_SHORT_PRON

            out_words.append(w)
            out_prons.append(out_pron)
        else:
            out_words.append(w)
            out_prons.append(ps)

    # Fencepost
    out_words.append(words[-1])
    out_prons.append(prons[-1])

    return out_words, out_prons


def preprocess_prons(prons, words):
    """Complete pronunciations preprocessing. Strip consonants, strip stresses,
    disambiguate indefinite articles."""
    out = []

    _, fixed_prons = disambiguate_indefinite_article_prons(words, prons)

    for ps in fixed_prons:
        os = [strip_consonants(strip_stress(p)) for p in ps]
        # os = [['_W_'] + strip_consonants(strip_stress(p)) + ['_/W_'] for p in ps]
        out.append(os)

    # More processing steps:
    #  - disambiguate the
    return out


##############################################################################


def pron_bleu(prons_1, prons_2):
    """Returns the maximum BLEU score of two pronunciations."""
    logging.debug('Prons 1: {0}'.format(prons_1))
    logging.debug('Prons 2: {0}'.format(prons_2))

    best_score = -1.0
    best_p1 = None
    best_p2 = None
    for p1 in prons_1:
        for p2 in prons_2:
            if p1 == p2:
                bleu = 1.0
            else:
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

    prons = get_prons(words, prondict=prondict, unknown_pron_client=Pronounce())
    prons = preprocess_prons(prons, words=words)

    logging.debug('Words: {0}\nProns: {1}'.format(words, prons))

    # Your code goes here
    pair_scores = {}
    pair_prons = {}
    for ps1, w1 in itertools.izip(prons, words):
        for ps2, w2 in itertools.izip(prons, words):
            if w1 == w2:
                continue
            if ((w1, w2) in pair_scores) or ((w2, w1) in pair_scores):
                continue

            p1, p2, score = pair_score_fn(ps1, ps2)
            pair_scores[(w1, w2)] = score
            pair_prons[(w1, w2)] = (p1, p2)
            logging.debug('Scoring pair: {0} with {1:.3f}'.format((w1, w2), score))

    return pair_scores, pair_prons


##############################################################################

# Postprocessing - after the word scores have been computed


def rhyme_score_grid(words, prondict, pair_score_fn=pron_bleu):
    """Visualizes the pairwise rhyming scores."""
    pair_scores, pair_prons = word_rhyming_table(words, prondict=prondict, pair_score_fn=pair_score_fn)

    score_grid = numpy.zeros((len(words), len(words))) - 1.0

    for i1, w1 in enumerate(words):
        for i2, w2 in enumerate(words):
            # check if it's there
            key = w1, w2
            if (w1, w2) not in pair_scores:
                if (w2, w1) not in pair_scores:
                    continue
                else:
                    key = w2, w1

            s = pair_scores[key]
            score_grid[i1, i2] = s
            score_grid[i2, i1] = s

    return score_grid


def aggregate_score(grid):
    """Experiments with how to aggregate the score to get sensible results."""
    # Each word only needs to rhyme once.
    maxima = grid.max(axis=0)
    return numpy.average(maxima)

##############################################################################


def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__, add_help=True,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-t', '--text', action='store', nargs='+',
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

    words = tokenize(' '.join(args.text))

    # Your code goes here
    # pair_scores, pair_prons = word_rhyming_table(words, prondict=cmudict, pair_score_fn=pron_bleu)
    score_grid = rhyme_score_grid(words, prondict=cmudict, pair_score_fn=pron_bleu)

    final_score = aggregate_score(score_grid)
    print(final_score)


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
