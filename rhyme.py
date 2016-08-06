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

# We need a pronunciation substitution table.
#         AA	odd     AA D
#         AE	at	AE T
#         AH	hut	HH AH T
#         AO	ought	AO T
#         AW	cow	K AW
#         AY	hide	HH AY D
#         B 	be	B IY
#         CH	cheese	CH IY Z
#         D 	dee	D IY
#         DH	thee	DH IY
#         EH	Ed	EH D
#         ER	hurt	HH ER T
#         EY	ate	EY T
#         F 	fee	F IY
#         G 	green	G R IY N
#         HH	he	HH IY
#         IH	it	IH T
#         IY	eat	IY T
#         JH	gee	JH IY
#         K 	key	K IY
#         L 	lee	L IY
#         M 	me	M IY
#         N 	knee	N IY
#         NG	ping	P IH NG
#         OW	oat	OW T
#         OY	toy	T OY
#         P 	pee	P IY
#         R 	read	R IY D
#         S 	sea	S IY
#         SH	she	SH IY
#         T 	tea	T IY
#         TH	theta	TH EY T AH
#         UH	hood	HH UH D
#         UW	two	T UW
#         V 	vee	V IY
#         W 	we	W IY
#         Y 	yield	Y IY L D
#         Z 	zee	Z IY
#         ZH	seizure	S IY ZH ER

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


def pron_bleu(prons_1, prons_2):
    """Returns the maximum BLEU score of two pronunciations."""
    logging.debug('Prons 1: {0}'.format(prons_1))
    logging.debug('Prons 2: {0}'.format(prons_2))

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

    prons = get_prons(words, prondict=prondict, unknown_pron_client=Pronounce())

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

    return pair_scores, pair_prons


def line_rhymes(line_1, line_2, prondict=None, pair_score_fn=pron_bleu):
    """This function counts rhymes in line_1 vs line_2."""
    raise NotImplementedError()


##############################################################################


def rhyme_score(words, prondict, pair_score_fn=pron_bleu):
    """Computes the overall rhyming score. This is the "outward-facing"
    function that wraps all the pieces.

    Current implementation: compute BLEU for all word pairs.

    Needs some visualization... heatmap with cells! (meshgrid)
    """
    raise NotImplementedError()


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

    words = args.text

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
