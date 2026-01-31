# encoding: utf-8
"""
Models package.

This package contains pre-trained neural network models for madmom.

SECURITY NOTICE:
----------------
The models in this package are stored as Python pickle (.pkl) files.
Pickle files can execute arbitrary code when loaded, which is a security risk.

To mitigate this risk, madmom-modern provides:
1. SHA256 hash verification of all bundled models
2. A RestrictedUnpickler that only allows known-safe classes

Always use the secure loading functions provided by this module:
- `secure_load(filepath)` - Load with hash verification and restricted unpickling
- `load_model(path)` - Load bundled models by relative path

NEVER load pickle files from untrusted sources!

For more information, see:
- https://docs.python.org/3/library/pickle.html#restricting-globals
- https://huggingface.co/docs/hub/security-pickle

Please see the LICENSE file for licensing details of this package.
"""

from __future__ import absolute_import, division, print_function

import os
import glob

MODEL_PATH = os.path.dirname(__file__)

# Import secure loading utilities
from .secure_loader import (
    secure_load,
    load_model,
    verify_model_integrity,
    ModelIntegrityError,
    UnsafePickleError,
)


def models(pattern, path=MODEL_PATH):
    """
    Retrieve all models in path given a file name pattern.

    Parameters
    ----------
    pattern : str
        Pattern to be matched.
    path : str, optional
        Path to search for model files.

    Returns
    -------
    models : list
        Sorted list of matching model file names.

    """
    return sorted(glob.glob('%s/%s' % (path, pattern)))


# beats
BEATS_LSTM = models('beats/2016/beats_lstm_[1-8].pkl')
BEATS_BLSTM = models('beats/2015/beats_blstm_[1-8].pkl')
BEATS_TCN = models("beats/2019/beats_tcn_[1-8].pkl")

# downbeats
DOWNBEATS_BLSTM = models('downbeats/2016/downbeats_blstm_[1-8].pkl')
DOWNBEATS_BGRU = [models('downbeats/2016/downbeats_bgru_rhythmic_*.pkl'),
                  models('downbeats/2016/downbeats_bgru_harmonic_*.pkl')]
# notes
NOTES_BRNN = models('notes/2013/notes_brnn.pkl')
NOTES_CNN = models('notes/2019/notes_cnn.pkl')
NOTES_CNN_MIREX = models('notes/2018/notes_cnn_[12].pkl')

# onsets
ONSETS_RNN = models('onsets/2013/onsets_rnn_[1-8].pkl')
ONSETS_BRNN = models('onsets/2013/onsets_brnn_[1-8].pkl')
ONSETS_BRNN_PP = models('onsets/2014/onsets_brnn_pp_[1-8].pkl')
ONSETS_CNN = models('onsets/2013/onsets_cnn.pkl')

# patterns
PATTERNS_BALLROOM = models('patterns/2013/ballroom_pattern_[34]_4.pkl')

# chroma
CHROMA_DNN = models('chroma/2016/chroma_dnn.pkl')

# chords
CHORDS_DCCRF = models('chords/2016/chords_dccrf.pkl')
CHORDS_CNN_FEAT = models('chords/2016/chords_cnnfeat.pkl')
CHORDS_CFCRF = models('chords/2016/chords_cnncrf.pkl')

# key
KEY_CNN = models('key/2018/key_cnn.pkl')

# keep namespace clean
del os, glob
