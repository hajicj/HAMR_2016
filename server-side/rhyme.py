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

import networkx

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

# Words


def filter_stopwords(words):
    stop = set(nltk.corpus.stopwords.words('english'))
    return [w for w in words if w not in stop]


STOPWORDS = {u'its', u'before', u'o', u'hadn',
             u'herself', u'll', u'had', u'should', u'to', u'only', u'ours', u'has', u'do', u'them', u'this',
             u'his', u'very', u'they', u'not', u'during', u'now', u'him', u'nor', u'd', u'did', u'didn',
             u'she', u'each', u'further', u'where', u'few', u'because', u'doing', u'some', u'hasn', u'are', u'our',
             u'ourselves', u'out', u'what', u'for', u'while', u're', u'does', u'above', u'between', u'mustn', u't',
             u'be', u'we', u'who', u'were', u'here', u'shouldn', u'hers', u'by', u'on', u'about', u'couldn', u'of',
             u'against', u's', u'isn', u'or', u'own', u'into', u'yourself', u'down', u'mightn', u'wasn', u'your',
             u'from', u'her', u'their', u'aren', u'there', u'been', u'whom', u'too', u'wouldn', u'themselves', u'weren',
             u'was', u'until', u'more', u'that', u'but', u'don', u'with', u'than', u'those', u'he', u'me',
             u'myself', u'ma', u'these', u'up', u'will', u'below', u'ain', u'can', u'theirs', u'my', u'and', u've',
             u'then', u'is', u'am', u'it', u'doesn', u'an', u'as', u'itself', u'at', u'have', u'in', u'any', u'if',
             u'again', u'no', u'when', u'same', u'how', u'other', u'which', u'you', u'shan', u'needn', u'haven',
             u'after', u'most', u'such', u'why', u'a', u'off', u'i', u'm', u'yours', u'so', u'y', u'the', u'having',
             u'once'}
STRONG_STOPWORDS = {u'its', u'o',
                    u'll', u'had', u'to', u'has', u'do',
                    u'his', u'not', u'him', u'nor', u'd', u'did', u'didn',
                    u'she', u'each', u'doing', u'hasn', u'are', u'our',
                    u'for', u're', u'does', u'mustn', u't',
                    u'be', u'we', u'were', u'hers', u'by', u'on', u'couldn', u'of',
                    u's', u'isn', u'or', u'mightn', u'wasn', u'your',
                    u'from', u'her', u'their', u'aren', u'been', u'too', u'wouldn', u'weren',
                    u'was', u'but', u'don', u'with', u'than', u'he', u'me',
                    u'up', u'will', u'ain', u'can', u'my', u'and', u've',
                    u'is', u'am', u'it', u'doesn', u'an', u'as', u'at', u'have', u'in', u'if',
                    u'when', u'how', u'which', u'you', u'haven',
                    u'most', u'such', u'why', u'a', u'i', u'm', u'so', u'y', u'the',
                    }


def is_stopword(w):
    return w in STOPWORDS


def is_strong_stopword(w):
    return w in STRONG_STOPWORDS


def tokenize(text):
    tokens = nltk.tokenize.wordpunct_tokenize(text.translate({ord(c): None for c in string.punctuation}).lower())
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
            # The output format of the unknown pron client is a single-key dict...
            # The value is a list of prons. One of these is just the capitalized word.
            ps_dict = unknown_pron_client.p(add_fake_stress=True)
            ps = [v.split() for v in ps_dict.values()[0] if v.split()[0] in ARPAbet.phonemes]
            unknown_pron_client.words = []
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
        try:
            if p[-1] in '0123':
                out.append(p[:-1])
        except IndexError:
            print('Strange phoneme {0} in pron {1}'.format(p, pron))
            raise
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
    logging.debug('Fixed prons: {0}'.format(fixed_prons))

    for ps, w in zip(fixed_prons, words):
        try:
            os = [strip_consonants(strip_stress(p)) for p in ps]
        except IndexError:
            print('In pron/word: {0}/{1}'.format(ps, w))
            raise
        # os = [['_W_'] + strip_consonants(strip_stress(p)) + ['_/W_'] for p in ps]
        out.append(os)

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
            elif len(p1) == 0:
                logging.warn('Zero-length pron!')
                bleu = 0.0
            elif len(p2) == 0:
                logging.warn('Zero-length pron!')
                bleu = 0.0
            else:
                bleu = nltk.translate.bleu_score.sentence_bleu([p2], p1)
            if bleu > best_score:
                best_p1 = p1
                best_p2 = p2
                best_score = bleu

    return best_p1, best_p2, best_score


##############################################################################


def word_rhyming_table(words, prondict=None, pair_score_fn=pron_bleu, term_weights=None,
                       MAX_DISTANCE=16):

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
    for i1, (ps1, w1) in enumerate(itertools.izip(prons, words)):

        # Relevant window: either the first MAX_DISTANCE words, or the MAX_DISTANCE / 2 words either way,
        # or the last MAX_DISTANCE words.
        if i1 < (MAX_DISTANCE / 2):
            lower_lim, upper_lim = 0, min(MAX_DISTANCE, len(words))
        elif i1 > (len(words) - MAX_DISTANCE / 2):
            lower_lim, upper_lim = max(0, len(words) - MAX_DISTANCE), len(words)
        else:
            lower_lim, upper_lim = max(0, (i1 - MAX_DISTANCE / 2)), min((i1 + MAX_DISTANCE / 2), len(words))

        # print('Window for i1={0}: {1} -- {2}'.format(i1, lower_lim, upper_lim))

        for i2, (ps2, w2) in enumerate(itertools.izip(prons[lower_lim:upper_lim],
                                                      words[lower_lim:upper_lim])):
            if w1 == w2:
                continue
            if ((w1, w2) in pair_scores) or ((w2, w1) in pair_scores):
                continue
            if is_stopword(w1) and is_stopword(w2):
                continue
            if is_strong_stopword(w1) or is_strong_stopword(w2):
                continue

            p1, p2, score = pair_score_fn(ps1, ps2)

            # Weights of scores
            if term_weights is not None:
                weight1 = 1.0
                if w1 in term_weights:
                    weight1 = term_weights[w1]
                weight2 = 1.0
                if w2 in term_weights:
                    weight2 = term_weights[w2]
                mean_weight = numpy.sqrt(weight1 * weight2)
                score *= mean_weight

            pair_scores[(w1, w2)] = score
            pair_prons[(w1, w2)] = (p1, p2)
            logging.debug('Scoring pair: {0} with {1:.3f}'.format((w1, w2), score))

    return pair_scores, pair_prons


##############################################################################

# Postprocessing - after the word scores have been computed


def rhyme_score_grid(words, prondict, pair_score_fn=pron_bleu, nonnegative=False):
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

    if nonnegative is True:
        score_grid[score_grid < 0.0] = 0.0

    return score_grid


def aggregate_score(grid):
    """Experiments with how to aggregate the score to get sensible results."""
    # Each word only needs to rhyme once.
    maxima = grid.max(axis=0)
    return numpy.average(maxima)


def binarize_grid(grid, thr=0.8):

    binary_grid = grid * 1
    binary_grid[grid <= thr] = 0
    binary_grid[grid > thr] = 1
    for i in xrange(binary_grid.shape[0]):
        binary_grid[i, i] = 1
    return binary_grid


def find_rhyme_groups(grid, words, thr=0.8):
    """Identify groups of words that look like they're on a grid.
    These are cliques in an adjacency graph.
    """
    word_keys = [w + '_{0}'.format(i) for i, w in enumerate(words)]
    binary_grid = binarize_grid(grid, thr=thr)
    G = networkx.Graph()
    G.add_nodes_from(word_keys)
    for i, x in enumerate(word_keys):
        for j, y in enumerate(word_keys):
            if j < i:
                continue
            if binary_grid[i, j] > 0:
                G.add_edge(x, y)

    cliques = list(networkx.enumerate_all_cliques(G))
    return cliques


def multi_clique_ratio(cliques, min_level=3):
    multi_cliques = [c for c in cliques if len(c) >= min_level]
    return multi_cliques


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
    if __name__ == '__main__':
        print(final_score)


    _end_time = time.clock()
    logging.info('[XXXX] done in {0:.3f} s'.format(_end_time - _start_time))

    return final_score


def get_score(args_dict):
    """Call this from the judge. The args_dict should have the 'text' key:

    >>> args = {'text': ['This', 'is', 'some', 'rap', 'it', 'is', 'pretty', 'crap']}
    >>> score = get_score(args)

    """
    logging.info('Starting main...')
    _start_time = time.clock()

    cmudict = collections.defaultdict(list)
    for word, syl in nltk.corpus.cmudict.entries():
        cmudict[word].append(syl)

    words = tokenize(' '.join(args_dict['text']))

    # Your code goes here
    # pair_scores, pair_prons = word_rhyming_table(words, prondict=cmudict, pair_score_fn=pron_bleu)
    score_grid = rhyme_score_grid(words, prondict=cmudict, pair_score_fn=pron_bleu)
    final_score = aggregate_score(score_grid)

    _end_time = time.clock()
    logging.info('[XXXX] done in {0:.3f} s'.format(_end_time - _start_time))

    return final_score


##############################################################################


if __name__ == '__main__':
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    if args.debug:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

    main(args)
