"""
Secure Model Loader for madmom

This module provides secure loading of pickle model files with:
1. SHA256 hash verification to detect tampering
2. Restricted unpickler to limit code execution risks
3. Allowlisting of known-safe modules and classes

SECURITY WARNING:
Pickle files can execute arbitrary code when loaded. This secure loader
mitigates risks but cannot guarantee complete safety. Only load models
from trusted sources.

For more information on pickle security risks, see:
- https://docs.python.org/3/library/pickle.html#restricting-globals
- https://snyk.io/articles/python-pickle-poisoning-and-backdooring-pth-files/
"""

import hashlib
import io
import json
import pickle
import warnings
from pathlib import Path
from typing import Any, Optional, Set

# Directory containing this file
MODELS_DIR = Path(__file__).parent

# Allowlisted modules and classes that are safe to unpickle
# These are the modules used by madmom's neural network models
SAFE_MODULES: Set[str] = {
    'numpy',
    'numpy.core.multiarray',
    'numpy.core.numeric',
    'numpy._core.multiarray',
    'numpy._core.numeric',
    'scipy.sparse',
    'scipy.sparse._csr',
    'scipy.sparse.csr',
    'collections',
    'builtins',
    # madmom modules
    'madmom.ml.nn',
    'madmom.ml.nn.layers',
    'madmom.ml.nn.activations',
    'madmom.ml.hmm',
    'madmom.ml.crf',
    'madmom.features.beats_hmm',
}

# Specific classes that are allowed
SAFE_CLASSES: Set[str] = {
    # NumPy
    'numpy.ndarray',
    'numpy.dtype',
    'numpy.core.multiarray._reconstruct',
    'numpy.core.multiarray.scalar',
    'numpy._core.multiarray._reconstruct',
    'numpy._core.multiarray.scalar',
    # SciPy sparse
    'scipy.sparse._csr.csr_matrix',
    'scipy.sparse.csr.csr_matrix',
    # Built-in types
    'builtins.dict',
    'builtins.list',
    'builtins.tuple',
    'builtins.set',
    'builtins.frozenset',
    'builtins.bytes',
    'builtins.bytearray',
    'collections.OrderedDict',
    # madmom classes (layers)
    'madmom.ml.nn.layers.FeedForwardLayer',
    'madmom.ml.nn.layers.RecurrentLayer',
    'madmom.ml.nn.layers.BidirectionalLayer',
    'madmom.ml.nn.layers.ConvolutionalLayer',
    'madmom.ml.nn.layers.StrideLayer',
    'madmom.ml.nn.layers.MaxPoolLayer',
    'madmom.ml.nn.layers.BatchNormLayer',
    'madmom.ml.nn.layers.AverageLayer',
    'madmom.ml.nn.layers.LSTMLayer',
    'madmom.ml.nn.layers.GRULayer',
    'madmom.ml.nn.layers.TransposedConvolutionalLayer',
    'madmom.ml.nn.layers.PadLayer',
    # madmom activations
    'madmom.ml.nn.activations.Activation',
    'madmom.ml.nn.activations.sigmoid',
    'madmom.ml.nn.activations.tanh',
    'madmom.ml.nn.activations.relu',
    'madmom.ml.nn.activations.elu',
    'madmom.ml.nn.activations.softmax',
    'madmom.ml.nn.activations.linear',
}


class ModelIntegrityError(Exception):
    """Raised when a model file fails integrity verification."""
    pass


class UnsafePickleError(Exception):
    """Raised when a pickle file attempts to load unsafe objects."""
    pass


class RestrictedUnpickler(pickle.Unpickler):
    """
    A restricted unpickler that only allows loading of known-safe classes.

    This prevents arbitrary code execution by blocking unknown module/class
    combinations during unpickling.
    """

    def find_class(self, module: str, name: str) -> Any:
        """
        Override find_class to restrict which classes can be instantiated.

        Parameters
        ----------
        module : str
            The module name.
        name : str
            The class/function name.

        Returns
        -------
        Any
            The class or function if it's allowed.

        Raises
        ------
        UnsafePickleError
            If the module/class combination is not in the allowlist.
        """
        full_name = f"{module}.{name}"

        # Check if this specific class is allowed
        if full_name in SAFE_CLASSES:
            return super().find_class(module, name)

        # Check if the module is in safe modules and the class looks safe
        if module in SAFE_MODULES:
            # Additional check: block anything that looks like code execution
            dangerous_names = {'exec', 'eval', 'compile', 'open', 'input',
                             '__import__', 'getattr', 'setattr', 'delattr',
                             'globals', 'locals', '__builtins__', 'system',
                             'popen', 'subprocess', 'os', 'sys'}
            if name.lower() not in dangerous_names:
                return super().find_class(module, name)

        raise UnsafePickleError(
            f"Blocked unsafe pickle class: {full_name}\n"
            f"This class is not in the allowlist for safe model loading.\n"
            f"If this is a legitimate madmom class, please report this issue."
        )


def compute_file_hash(filepath: Path) -> str:
    """
    Compute SHA256 hash of a file.

    Parameters
    ----------
    filepath : Path
        Path to the file.

    Returns
    -------
    str
        Hex-encoded SHA256 hash.
    """
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_manifest() -> dict:
    """
    Load the model manifest containing expected hashes.

    Returns
    -------
    dict
        Dictionary mapping model paths to their expected SHA256 hashes.
    """
    manifest_path = MODELS_DIR / 'model_manifest.json'
    if not manifest_path.exists():
        warnings.warn(
            "Model manifest not found. Hash verification disabled.\n"
            "This reduces security - consider reinstalling madmom-modern.",
            UserWarning
        )
        return {}

    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    return manifest.get('models', {})


def verify_model_integrity(filepath: Path, manifest: Optional[dict] = None) -> bool:
    """
    Verify a model file's integrity using SHA256 hash.

    Parameters
    ----------
    filepath : Path
        Path to the model file.
    manifest : dict, optional
        Model manifest dictionary. If None, loads from default location.

    Returns
    -------
    bool
        True if the file passes verification.

    Raises
    ------
    ModelIntegrityError
        If the file's hash doesn't match the expected value.
    """
    if manifest is None:
        manifest = load_manifest()

    if not manifest:
        return True  # No manifest, skip verification (with warning already issued)

    # Get relative path from models directory
    try:
        rel_path = filepath.relative_to(MODELS_DIR)
        rel_path_str = str(rel_path).replace('\\', '/')  # Normalize for Windows
    except ValueError:
        # File is not in models directory
        warnings.warn(
            f"Model file {filepath} is outside the models directory.\n"
            "Hash verification skipped. Only load models from trusted sources!",
            UserWarning
        )
        return True

    expected_hash = manifest.get(rel_path_str)
    if expected_hash is None:
        warnings.warn(
            f"Model file {rel_path_str} not found in manifest.\n"
            "Hash verification skipped. This file may have been added after installation.",
            UserWarning
        )
        return True

    actual_hash = compute_file_hash(filepath)

    if actual_hash != expected_hash:
        raise ModelIntegrityError(
            f"Model file integrity check FAILED: {filepath}\n"
            f"Expected hash: {expected_hash}\n"
            f"Actual hash:   {actual_hash}\n"
            f"\n"
            f"This file may have been tampered with or corrupted.\n"
            f"DO NOT LOAD THIS FILE - it could contain malicious code.\n"
            f"Please reinstall madmom-modern from a trusted source."
        )

    return True


def secure_load(filepath: Path, verify_hash: bool = True) -> Any:
    """
    Securely load a pickle model file with integrity verification.

    This function:
    1. Verifies the file's SHA256 hash against the known-good manifest
    2. Uses a restricted unpickler to block unsafe classes
    3. Only allows known madmom model classes to be loaded

    Parameters
    ----------
    filepath : Path
        Path to the pickle file.
    verify_hash : bool, optional
        Whether to verify the file's hash. Default True.
        Setting to False is STRONGLY DISCOURAGED.

    Returns
    -------
    Any
        The unpickled object.

    Raises
    ------
    ModelIntegrityError
        If hash verification fails.
    UnsafePickleError
        If the pickle contains unsafe objects.
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")

    # Step 1: Verify integrity
    if verify_hash:
        verify_model_integrity(filepath)
    else:
        warnings.warn(
            "Hash verification disabled! This is a security risk.\n"
            "Only disable this if you trust the source of the model file.",
            UserWarning
        )

    # Step 2: Load with restricted unpickler
    with open(filepath, 'rb') as f:
        try:
            return RestrictedUnpickler(f).load()
        except UnsafePickleError:
            raise
        except Exception as e:
            raise UnsafePickleError(
                f"Failed to load model file: {filepath}\n"
                f"Error: {e}\n"
                f"This may indicate a corrupted or incompatible model file."
            ) from e


def load_model(model_path: str, verify_hash: bool = True) -> Any:
    """
    Load a madmom model by its relative path within the models directory.

    This is the recommended way to load bundled madmom models.

    Parameters
    ----------
    model_path : str
        Relative path to the model within the models directory.
        Example: "beats/2016/beats_lstm_1.pkl"
    verify_hash : bool, optional
        Whether to verify the file's hash. Default True.

    Returns
    -------
    Any
        The loaded model object.

    Examples
    --------
    >>> model = load_model("beats/2016/beats_lstm_1.pkl")
    """
    filepath = MODELS_DIR / model_path
    return secure_load(filepath, verify_hash=verify_hash)


# For backwards compatibility, also expose standard pickle load with warnings
def unsafe_load(filepath: Path) -> Any:
    """
    Load a pickle file WITHOUT security checks.

    WARNING: This function is UNSAFE and should only be used for debugging
    or when you completely trust the source of the file.

    Parameters
    ----------
    filepath : Path
        Path to the pickle file.

    Returns
    -------
    Any
        The unpickled object.
    """
    warnings.warn(
        "Using unsafe_load() bypasses all security checks!\n"
        "Only use this if you completely trust the source of the file.\n"
        "Consider using secure_load() instead.",
        UserWarning
    )
    with open(filepath, 'rb') as f:
        return pickle.load(f)
