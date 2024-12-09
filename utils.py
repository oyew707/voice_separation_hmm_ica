"""
-------------------------------------------------------
[Program Description]
-------------------------------------------------------
Author:  einsteinoyewole
Email:   [your email address]
-------------------------------------------------------
"""

# Imports
import os
import shutil

# Constants
mix_module = "/Users/einsteinoyewole/PycharmProjects/pywsj0-mix/generate_wsjmix.py"


def generate_mix_sounds(dry_run=True, sources=[2, 3, 4], frequencies=[8000, 16000], wsj0_path="wsj0",
                        ouput_path_prefix="wsj0-mix-{nsrc}mix-{sampling_freq}Hz"):
    """
    -------------------------------------------------------
    Generate the mix sounds for the given sources and frequencies
    -------------------------------------------------------
    Parameters:
         dry_run (bool) - If True, the function will not run the command to generate the mix sounds
         sources (list) - List of the number of sources to generate the mix sounds for
         frequencies (list) - List of the sampling frequencies to generate the mix sounds for
         wsj0_path (str) - Path to the wsj0 dataset
         ouput_path_prefix (str) - Prefix for the output path for the mix sounds
    -------------------------------------------------------
    """
    # Input Validation
    assert len(sources) > 0, "The list of sources must not be empty"
    assert len(frequencies) > 0, "The list of frequencies must not be empty"
    assert os.path.exists(wsj0_path), "The path to the wsj0 dataset does not exist"

    # Generate the mix sounds
    for nsrc in sources:
        for sampling_freq in frequencies:
            output_path = ouput_path_prefix.format(nsrc=nsrc, sampling_freq=sampling_freq)
            generate_mix_command = f"python3.13 {mix_module} -p {wsj0_path} -o {output_path} -n {nsrc} -sr {sampling_freq}"
            print('Running command:', generate_mix_command)
            if not dry_run:
                os.system(generate_mix_command)
    print('Done!')


def move_wv_files(src, dest, dry_run=True):
    """
    -------------------------------------------------------
    Move files from the source to the destination
    -------------------------------------------------------
    Parameters:
         src (str) - Source directory
         dest (str) - Destination directory
         dry_run (bool) - If True, the function will not move the files
    -------------------------------------------------------
    """
    os.makedirs(dest, exist_ok=True)
    for path, subdirs, files in os.walk(src):
        new_p = os.path.relpath(path, src)
        for name in files:
            src_file = os.path.join(path, name)
            dest_file = os.path.join(dest, new_p, name)
            if os.path.exists(dest_file):
                print(f"File {dest_file} already exists. Skipping...")
                continue
            if not dry_run:
                shutil.copy(src_file, dest_file)
            else:
                print(f"Moving {src_file} to {dest_file}")


def convert_wv_files_to_wav(dir, dry_run=True):
    """
    -------------------------------------------------------
    Convert wv files to wav files
    -------------------------------------------------------
    Parameters:
         dir (str) - directory containing the wv files
         dry_run (bool) - If True, the function will not convert the files
    -------------------------------------------------------
    """
    assert os.path.exists(dir), "The directory does not exist"

    for path, subdirs, files in os.walk(dir):
        for name in files:
            src_file = os.path.join(path, name)
            file_base, file_ext = os.path.splitext(src_file)
            if not file_ext.startswith('.wv'):
                print('Skipping file:', src_file)
                continue
            dest_file = file_base + '.wav'
            if os.path.exists(dest_file):
                print(f"File {dest_file} already exists. Skipping...")
                continue
            command = f"ffmpeg -i {src_file} {dest_file} -loglevel error -n"
            if not dry_run:
                os.system(command)
            else:
                print(f"Converting {src_file} to {dest_file}: {command=}")
    print('Done!')
