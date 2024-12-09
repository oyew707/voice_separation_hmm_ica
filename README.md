# Hidden Markov Independent Component Analysis (HMICA)

A Python implementation of Hidden Markov Independent Component Analysis described in [Penny et al. (2000)](https://doi.org/10.1007/978-1-4471-0443-8_1). This implementation supports both standard ICA and Generalized Autoregressive (GAR) source modeling.

## Installation

Install required packages using pip:
```bash
pip install -r requirements.txt
```

Required packages:
- tensorflow >= 2.12.0
- numpy >= 1.23.5
- scipy >= 1.10.1
- scikit-learn >= 1.2.2
- tqdm >= 4.65.0
- numpy>=1.26.4
- soundfile>=0.12.1
- fast-bss-eval>=0.1.3

## Usage

### Basic Example

```python
from HMICA import HMICA
import numpy as np

# Initialize model
model = HMICA(
    n_states=2,          # Number of HMM states
    n_sources=2,         # Number of sources to extract
    whiten=True,         # Whether to whiten input data
    use_gar=True,        # Use Generalized AutoRegressive modeling
    gar_order=2,         # Order of GAR model
    random_state=42      # For reproducibility
)

# Fit and transform data
X = np.random.randn(1000, 2)  # Your mixed signals
result = model.fit_transform(X)

# Access separated sources and state sequence
sources = result['source_signals']      # Shape: [n_samples, n_sources]
states = result['state_sequence']       # Shape: [n_samples, n_states]

# Reconstruct original signals
X_reconstructed = model.inverse_transform(sources, states)
```

### Advanced Configuration

```python
model = HMICA(
    n_states=2,
    n_sources=2,
    whiten=True,
    use_gar=True,
    gar_order=2,
    inference_method='viterbi',  # 'viterbi' or 'smoothing'
    learning_rates={            # Learning rates for different parameters
        'W': 1e-4,             # Unmixing matrix
        'R': 1e-4,             # Shape parameter
        'beta': 1e-4,          # Scale parameter
        'C': 1e-4              # GAR coefficients
    },
    max_iter={                 # Maximum iterations
        'hmm': 100,           # HMM optimization
        'ica': 1000           # ICA optimization
    },
    tol={                     # Convergence tolerances
        'hmm': 1e-4,         # HMM optimization
        'ica': 1e-2          # ICA optimization
    },
    optimizer_patience=5,     # Early stopping patience
    warmup_period=5,         # Initial iterations before convergence check
    n_processes=None,        # Number of parallel processes (None=single process)
    random_state=42
)
```

## Evaluation Metrics

The package includes two evaluation metrics:

```python
from evaluation import signal_to_noise_ratio, signal_to_distortion_ratio

# Calculate SNR
snr = signal_to_noise_ratio(original_sources, separated_sources)

# Calculate SDR
sdr = signal_to_distortion_ratio(original_sources, separated_sources)
```

## Audio Processing Example mixing: WSJ0 dataset
- Data Source: [kaggle link](https://www.kaggle.com/datasets/stsword/wsj0original/discussion?sort=hotness)
- Clone Mixing scrip: `git clone pywsj0-mix`
- unzip the data: `unzip archive.zip`
- Consolidate all the data from different discs (the zipped data is separted by 15 discs)
   ```python3
  from utils import move_wv_files
    src_dir = [
        "/Users/einsteinoyewole/Downloads/csr_1/11-1.1/wsj0",
                "/Users/einsteinoyewole/Downloads/csr_1/11-2.1/wsj0",
                "/Users/einsteinoyewole/Downloads/csr_1/11-3.1/wsj0",
                "/Users/einsteinoyewole/Downloads/csr_1/11-4.1/wsj0",
                "/Users/einsteinoyewole/Downloads/csr_1/11-5.1/wsj0",
                "/Users/einsteinoyewole/Downloads/csr_1/11-6.1/wsj0",
                "/Users/einsteinoyewole/Downloads/csr_1/11-7.1/wsj0",
                "/Users/einsteinoyewole/Downloads/csr_1/11-8.1/wsj0",
    "/Users/einsteinoyewole/Downloads/csr_1/11-9.1/wsj0",
    "/Users/einsteinoyewole/Downloads/csr_1/11-10.1/wsj0",
    "/Users/einsteinoyewole/Downloads/csr_1/11-11.1/wsj0",
    "/Users/einsteinoyewole/Downloads/csr_1/11-12.1/wsj0",
    "/Users/einsteinoyewole/Downloads/csr_1/11-13.1/wsj0",
    "/Users/einsteinoyewole/Downloads/csr_1/11-14.1/wsj0",
    "/Users/einsteinoyewole/Downloads/csr_1/11-15.1/wsj0",
               ]
    
    dest_dir_prefix = "/Users/einsteinoyewole/PycharmProjects/ds-ga1018_project/code/data/wsj0"
    for src in src_dir:
         # use dry_run=False to move the files
         move_wv_files(src, dest_dir_prefix, dry_run=True)
    ```

- Convert from wv files to wav files
    ```python3
    from utils import convert_wv_to_wav
    src_dir = "/Users/einsteinoyewole/PycharmProjects/ds-ga1018_project/code/data/wsj0"
    # Use dry_run=False to convert the files
    convert_wv_to_wav(src_dir, dry_run=True)
    ```
- Mix the audio files
    ```python3
    from utils import generate_mix_sounds
    src_dir = "/Users/einsteinoyewole/PycharmProjects/ds-ga1018_project/code/data/wsj0"
    output_prefix = "/Users/einsteinoyewole/PycharmProjects/ds-ga1018_project/code/mix/{nsrc}_speaker_{sampling_freq}_hz/"
    # Use dry_run=False to mix the files
    generate_mix_sounds(src_dir, output_prefix, dry_run=True)
    ```