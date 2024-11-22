# Voice Separation using Hidden Markov Independent Component analysis 

Document: [Link](https://docs.google.com/document/d/1BUl8Jnd-v2zFqgMAEvz9YW2oHm_f9Vh3z6mTCJG5L0M/edit?usp=sharing)

## Audio mixing: WSJ0 dataset
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