# Security Policy

## Pickle File Security

### The Risk

madmom uses Python pickle (`.pkl`) files to store pre-trained neural network models. **Pickle files are inherently dangerous** because they can execute arbitrary code when loaded.

From [Python's official documentation](https://docs.python.org/3/library/pickle.html):

> **Warning:** The pickle module is not secure. Only unpickle data you trust.
> It is possible to construct malicious pickle data which will execute arbitrary code during unpickling.

### Why This Matters

If you load a malicious pickle file, an attacker could:
- Install malware or backdoors on your system
- Steal sensitive data including credentials and API keys
- Gain full control of your machine or server
- Use your system to attack others

According to [Snyk research](https://snyk.io/articles/python-pickle-poisoning-and-backdooring-pth-files/), 83.5% of ML models on platforms like HuggingFace use pickle-based formats, making this a widespread concern.

### Bundled Model Verification

All 92 pre-trained model files bundled with madmom-modern have been security scanned using:
- [Fickling](https://github.com/trailofbits/fickling) - Trail of Bits' pickle decompiler and analyzer
- Byte-level pattern matching for dangerous imports

**Scan Results:**
- No dangerous module imports detected (no `os`, `subprocess`, `socket`, etc.)
- Only expected imports found: `numpy`, `scipy.sparse`, `madmom.ml.nn.*`, `copy_reg`
- All files contain only neural network weights and layer definitions

The models originate from the official [CPJKU/madmom_models](https://github.com/CPJKU/madmom_models) repository.

### How madmom-modern Mitigates This Risk

We implement multiple layers of defense:

#### 1. SHA256 Hash Verification

All bundled model files have their SHA256 hashes recorded in `model_manifest.json`. Before loading any model, the secure loader verifies the file hasn't been tampered with:

```python
from madmom.models import secure_load, ModelIntegrityError

try:
    model = secure_load("/path/to/model.pkl")
except ModelIntegrityError as e:
    print(f"Model file may be compromised: {e}")
```

#### 2. Restricted Unpickler

Our `RestrictedUnpickler` only allows loading of known-safe classes:
- NumPy arrays and data types
- SciPy sparse matrices
- madmom neural network layer classes
- Standard Python containers (dict, list, tuple)

Any attempt to load unknown or dangerous classes raises an `UnsafePickleError`:

```python
from madmom.models import UnsafePickleError

try:
    model = secure_load(untrusted_file)
except UnsafePickleError as e:
    print(f"Blocked unsafe pickle operation: {e}")
```

#### 3. Module Allowlisting

Only specific modules can be imported during unpickling:
- `numpy`, `scipy.sparse`
- `madmom.ml.nn`, `madmom.ml.nn.layers`, `madmom.ml.nn.activations`
- `collections`, `builtins`

### Best Practices

1. **Only load models from trusted sources**
   - Use the bundled models that ship with madmom-modern
   - Verify third-party models come from reputable sources

2. **Always use secure loading**
   ```python
   from madmom.models import secure_load
   model = secure_load("path/to/model.pkl")
   ```

3. **Never disable hash verification** without understanding the risks
   ```python
   # DON'T DO THIS unless you absolutely must
   model = secure_load("path/to/model.pkl", verify_hash=False)
   ```

4. **Treat .pkl files like executables**
   - Don't download them from untrusted websites
   - Don't open email attachments containing pickle files
   - Scan suspicious files in isolated environments

5. **Consider safer alternatives for new models**
   - [SafeTensors](https://huggingface.co/docs/safetensors) - cannot execute code
   - NumPy's `.npz` format - for simple weight arrays
   - ONNX - for portable model exchange

### Reporting Security Issues

If you discover a security vulnerability in madmom-modern:

1. **Do not** open a public GitHub issue
2. Email the maintainers directly with details
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Any suggested fixes

We will respond within 48 hours and work to address the issue promptly.

### Updating Model Hashes

If you need to regenerate the model manifest (e.g., after updating models):

```bash
cd madmom-modern
find madmom/models -name "*.pkl" -exec sha256sum {} \; | sort -k2 > hashes.txt
```

Then update `madmom/models/model_manifest.json` accordingly.

### References

- [Python pickle security](https://docs.python.org/3/library/pickle.html#restricting-globals)
- [Snyk: Python Pickle Poisoning](https://snyk.io/articles/python-pickle-poisoning-and-backdooring-pth-files/)
- [HuggingFace Pickle Scanning](https://huggingface.co/docs/hub/security-pickle)
- [SafeTensors: A Secure Alternative](https://dev.to/lukehinds/understanding-safetensors-a-secure-alternative-to-pickle-for-ml-models-o71)
