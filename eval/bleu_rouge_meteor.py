
def test_result(cad_file,ref_file,output_filePath):
    bleu_output = output_filePath + "/bleu_scores.txt"
    rouge_output = output_filePath + "/rouge_scores.txt"
    meteor_output = output_filePath + "/meteor_scores.txt"
    our_result = cad_file
    truth = ref_file
    with open(our_result, 'r',encoding='utf-8') as paper, open(truth, 'r',encoding='utf-8') as gold, open(bleu_output,'w') as bleu_result,open(rouge_output,'w') as rouge_result,open(meteor_output,'w') as meteor_result:
        fpaper = paper.read().splitlines()
        fpaper_get = []
        for item in fpaper:
            fpaper_get.append(item)

        # fgold = gold.read().splitlines()
        fgold_get = []
        fgold = gold.read().splitlines()
        for item in fgold:
            fgold_get.append(item)
        #print("bbb:",len(fgold_get))


        bleu, rouge_l, meteor = eval_accuracies(fpaper_get, fgold_get,bleu_output,rouge_output,meteor_output)
        print("bleu = {}".format(bleu))
        print("rouge-l = {}".format(rouge_l))
        print("meteor = {}".format(meteor))
    return None 


def eval_accuracies(model_generated, target_truth, bleu_output,rouge_output,meteor_output):
    generated = {k: [v.strip()] for k, v in enumerate(model_generated)}
    target_truth = {k: [v.strip()] for k, v in enumerate(target_truth)}
    assert sorted(generated.keys()) == sorted(target_truth.keys())

    print("bleu begin:")
    # Compute BLEU scores
    corpus_bleu_r, bleu, ind_bleu, bleu_4 = corpus_bleu(generated, target_truth, bleu_output)

    print("rouge begin:")
    # Compute ROUGE_L scores
    rouge_calculator = Rouge()
    rouge_l, ind_rouge = rouge_calculator.compute_score(target_truth, generated, rouge_output)

    print("meteor begin:")
    # Compute METEOR scores
    meteor_calculator = Meteor()
    meteor, _ = meteor_calculator.compute_score(target_truth, generated, meteor_output)


    return bleu * 100, rouge_l * 100, meteor * 100


# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Python implementation of BLEU and smooth-BLEU.
This module provides a Python implementation of BLEU and smooth-BLEU.
Smooth BLEU is computed following the method outlined in the paper:
Chin-Yew Lin, Franz Josef Och. ORANGE: a method for evaluating automatic
evaluation metrics for machine translation. COLING 2004.
"""

import collections
import math


def _get_ngrams(segment, max_order):
    """Extracts all n-grams upto a given maximum order from an input segment.
    Args:
      segment: text segment from which n-grams will be extracted.
      max_order: maximum length in tokens of the n-grams returned by this
          methods.
    Returns:
      The Counter containing all n-grams upto max_order in segment
      with a count of how many times each n-gram occurred.
    """
    ngram_counts = collections.Counter()
    for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
            ngram = tuple(segment[i:i + order])
            ngram_counts[ngram] += 1
    return ngram_counts


def compute_bleu(reference_corpus, translation_corpus, max_order=4,
                 smooth=False):
    """Computes BLEU score of translated segments against one or more references.
    Args:
      reference_corpus: list of lists of references for each translation. Each
          reference should be tokenized into a list of tokens.
      translation_corpus: list of translations to score. Each translation
          should be tokenized into a list of tokens.
      max_order: Maximum n-gram order to use when computing BLEU score.
      smooth: Whether or not to apply Lin et al. 2004 smoothing.
    Returns:
      3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram
      precisions and brevity penalty.
    """
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    reference_length = 0
    translation_length = 0
    for (references, translation) in zip(reference_corpus,
                                         translation_corpus):
        reference_length += min(len(r) for r in references)
        translation_length += len(translation)

        merged_ref_ngram_counts = collections.Counter()
        for reference in references:
            merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
        translation_ngram_counts = _get_ngrams(translation, max_order)
        overlap = translation_ngram_counts & merged_ref_ngram_counts
        #print(len(overlap))
        #print(len(translation_ngram_counts))
        for ngram in overlap:
            matches_by_order[len(ngram) - 1] += overlap[ngram]
        for order in range(1, max_order + 1):
            possible_matches = len(translation) - order + 1
            if possible_matches > 0:
                possible_matches_by_order[order - 1] += possible_matches

    precisions = [0] * max_order
    for i in range(0, max_order):
        if smooth:
            precisions[i] = ((matches_by_order[i] + 1.) /
                             (possible_matches_by_order[i] + 1.))
        else:
            if possible_matches_by_order[i] > 0:
                precisions[i] = (float(matches_by_order[i]) /
                                 possible_matches_by_order[i])
            else:
                precisions[i] = 0.0

    if min(precisions) > 0:
        p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0

    ratio = float(translation_length) / reference_length

    if ratio > 1.0:
        bp = 1.
    else:
        bp = math.exp(1 - 1. / ratio)
    #print("p_log_sum",p_log_sum)
    #print("geo_mean",geo_mean)
    #print("bp:",bp)
    bleu = geo_mean * bp

    return (bleu, precisions, bp, ratio, translation_length, reference_length)


def corpus_bleu(hypotheses, references, bleu_output):
    refs = []
    hyps = []
    count = 0
    total_score = 0.0

    assert (sorted(hypotheses.keys()) == sorted(references.keys()))
    Ids = list(hypotheses.keys())
    ind_score = dict()
    with open(bleu_output,'w') as bleu_result:
        for id in Ids:
            hyp = hypotheses[id][0].split()
            ref = [r.split() for r in references[id]]
            hyps.append(hyp)
            refs.append(ref)

            score = compute_bleu([ref], [hyp], smooth=True)[0]
            bleu_result.write(f"{score}\n")
            total_score += score
            count += 1
            ind_score[id] = score
            # if count % 1000 == 0:
            #     print(count)

    avg_score = total_score / count
    #corpus_bleu = compute_bleu(refs, hyps, smooth=True)[0]
    #bleu_4 = compute_bleu(refs, hyps, smooth=True)[1][3] * compute_bleu(refs, hyps, smooth=True)[2]
    bleu_4 = 1
    return corpus_bleu, avg_score, ind_score, bleu_4



#!/usr/bin/env python
#
# File Name : rouge.py
#
# Description : Computes ROUGE-L metric as described by Lin and Hovey (2004)
#
# Creation Date : 2015-01-07 06:03
# Author : Ramakrishna Vedantam <vrama91@vt.edu>

import numpy as np


def my_lcs(string, sub):
    """
    Calculates longest common subsequence for a pair of tokenized strings
    :param string : list of str : tokens from a string split using whitespace
    :param sub : list of str : shorter string, also split using whitespace
    :returns: length (list of int): length of the longest common subsequence between the two strings
    Note: my_lcs only gives length of the longest common subsequence, not the actual LCS
    """
    if len(string) < len(sub):
        sub, string = string, sub

    lengths = [[0 for i in range(0, len(sub) + 1)] for j in range(0, len(string) + 1)]

    for j in range(1, len(sub) + 1):
        for i in range(1, len(string) + 1):
            if string[i - 1] == sub[j - 1]:
                lengths[i][j] = lengths[i - 1][j - 1] + 1
            else:
                lengths[i][j] = max(lengths[i - 1][j], lengths[i][j - 1])

    return lengths[len(string)][len(sub)]


class Rouge():
    '''
    Class for computing ROUGE-L score for a set of candidate sentences for the MS COCO test set
    '''

    def __init__(self):
        # vrama91: updated the value below based on discussion with Hovey
        self.beta = 1.2

    def calc_score(self, candidate, refs):
        """
        Compute ROUGE-L score given one candidate and references for an image
        :param candidate: str : candidate sentence to be evaluated
        :param refs: list of str : COCO reference sentences for the particular image to be evaluated
        :returns score: int (ROUGE-L score for the candidate evaluated against references)
        """
        assert (len(candidate) == 1)
        assert (len(refs) > 0)
        prec = []
        rec = []

        # split into tokens
        token_c = candidate[0].split(" ")

        for reference in refs:
            # split into tokens
            token_r = reference.split(" ")
            # compute the longest common subsequence
            lcs = my_lcs(token_r, token_c)
            prec.append(lcs / float(len(token_c)))
            rec.append(lcs / float(len(token_r)))

        prec_max = max(prec)
        rec_max = max(rec)

        if prec_max != 0 and rec_max != 0:
            score = ((1 + self.beta ** 2) * prec_max * rec_max) / float(rec_max + self.beta ** 2 * prec_max)
        else:
            score = 0.0
        return score

    def compute_score(self, gts, res, rouge_output):
        """
        Computes Rouge-L score given a set of reference and candidate sentences for the dataset
        Invoked by evaluate_captions.py
        :param gts: dict : candidate / test sentences with "image name" key and "tokenized sentences" as values
        :param res: dict : reference MS-COCO sentences with "image name" key and "tokenized sentences" as values
        :returns: average_score: float (mean ROUGE-L score computed by averaging scores for all the images)
        """
        assert (sorted(gts.keys()) == sorted(res.keys()))
        imgIds = list(gts.keys())

        score = dict()
        with open(rouge_output,'w') as rouge_result:
            for id in imgIds:
                hypo = res[id]
                ref = gts[id]

                # Sanity check.
                assert (type(hypo) is list)
                assert (len(hypo) == 1)
                assert (type(ref) is list)
                assert (len(ref) > 0)

                score[id] = self.calc_score(hypo, ref)
                rouge_result.write(f"{score[id]}\n")

        average_score = np.mean(np.array(list(score.values())))
        return average_score, score

    def method(self):
        return "Rouge"



#!/usr/bin/env python

# Python wrapper for METEOR implementation, by Xinlei Chen
# Acknowledge Michael Denkowski for the generous discussion and help
# from __future__ import division

import atexit
import logging
import os
import subprocess
import sys
import threading

import psutil

# Assumes meteor-1.5.jar is in the same directory as meteor.py.  Change as needed.
METEOR_JAR = 'meteor-1.5.jar'


def enc(s):
    return s.encode('utf-8')


def dec(s):
    return s.decode('utf-8')


class Meteor:

    def __init__(self):
        # Used to guarantee thread safety
        self.lock = threading.Lock()

        mem = '2G'
        mem_available_G = psutil.virtual_memory().available / 1E9
        if mem_available_G < 2:
            logging.warning("There is less than 2GB of available memory.\n"
                            "Will try with limiting Meteor to 1GB of memory but this might cause issues.\n"
                            "If you have problems using Meteor, "
                            "then you can try to lower the `mem` variable in meteor.py")
            mem = '1G'

        meteor_cmd = ['java', '-jar', '-Xmx2G', METEOR_JAR, '-', '-', '-stdio', '-l', 'en', '-norm', '-a', 'data/paraphrase-en.gz']

        env = os.environ.copy()
        env['LC_ALL'] = "C"
        self.meteor_p = subprocess.Popen(meteor_cmd,
                                         cwd=os.path.dirname(os.path.abspath(__file__)),
                                         env=env,
                                         stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE)

        atexit.register(self.close)

    def close(self):
        with self.lock:
            if self.meteor_p:
                self.meteor_p.kill()
                self.meteor_p.wait()
                self.meteor_p = None
        # if the user calls close() manually, remove the
        # reference from atexit so the object can be garbage-collected.
        if atexit is not None and atexit.unregister is not None:
            atexit.unregister(self.close)

    def compute_score(self, gts, res, meteor_output):
        assert (gts.keys() == res.keys())
        imgIds = gts.keys()
        scores = []

        cnt = 0
        with self.lock:
            with open(meteor_output,'w') as meteor_result:
                for i in imgIds:
                    eval_line = 'EVAL'
                    #print(gts[i])
                    #print(res[i])
                    assert (len(res[i]) == 1)
                    stat = self._stat(res[i][0], gts[i])
                    eval_line += ' ||| {}'.format(stat)
                    self.meteor_p.stdin.write(enc('{}\n'.format(eval_line)))
                    self.meteor_p.stdin.flush()
                    v = self.meteor_p.stdout.readline()
                    meteor_result.write(f"{dec(v.strip())}\n")
                    cnt += 1
                    # if cnt % 1000 == 0:
                    #     print(cnt)

            # print(enc(eval_line))
            # self.meteor_p.stdin.write(enc('{}\n'.format(eval_line)))
            # self.meteor_p.stdin.flush()

            # for i in range(0, len(imgIds)):
            #     v = self.meteor_p.stdout.readline()
            #     print(dec(v.strip()))
            #     with open(meteor_output,'w') as meteor_result:
            #         meteor_result.write(f"{dec(v.strip())}\n")
            #     try:
            #         scores.append(float(dec(v.strip())))
            #     except:
            #         sys.stderr.write("Error handling value: {}\n".format(v))
            #         sys.stderr.write("Decoded value: {}\n".format(dec(v.strip())))
            #         sys.stderr.write("eval_line: {}\n".format(eval_line))
            #         # You can try uncommenting the next code line to show stderr from the Meteor JAR.
            #         # If the Meteor JAR is not writing to stderr, then the line will just hang.
            #         # sys.stderr.write("Error from Meteor:\n{}".format(self.meteor_p.stderr.read()))
            #         raise
            #score = float(dec(self.meteor_p.stdout.readline()).strip())
            score = 1

        return score, scores

    def method(self):
        return "METEOR"

    def _stat(self, hypothesis_str, reference_list):
        # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
        hypothesis_str = hypothesis_str.replace('|||', '').replace('  ', ' ')
        score_line = ' ||| '.join(('SCORE', ' ||| '.join(reference_list), hypothesis_str))
        #print(score_line)
        
        self.meteor_p.stdin.write(enc(score_line))
        self.meteor_p.stdin.write(enc('\n'))
        self.meteor_p.stdin.flush()

        #self.meteor_p.stdin.flush()
        return dec(self.meteor_p.stdout.readline()).strip()

    def _score(self, hypothesis_str, reference_list):
        with self.lock:
            # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
            hypothesis_str = hypothesis_str.replace('|||', '').replace('  ', ' ')
            score_line = ' ||| '.join(('SCORE', ' ||| '.join(reference_list), hypothesis_str))
            self.meteor_p.stdin.write(enc('{}\n'.format(score_line)))
            self.meteor_p.stdin.flush()
            stats = dec(self.meteor_p.stdout.readline()).strip()
            eval_line = 'EVAL ||| {}'.format(stats)
            # EVAL ||| stats 
            self.meteor_p.stdin.write(enc('{}\n'.format(eval_line)))
            self.meteor_p.stdin.flush()
            score = float(dec(self.meteor_p.stdout.readline()).strip())
            # bug fix: there are two values returned by the jar file, one average, and one all, so do it twice
            # thanks for Andrej for pointing this out
            score = float(dec(self.meteor_p.stdout.readline()).strip())
        return score

    def __del__(self):
        self.close()


if __name__ == "__main__":
    test_result('data/candidates','data/references','data')