
import os, shutil, re
import soundfile as sf
import argparse

def _create_dirs_given_dirnames(parent_dir_path, intermediate_dirs, dirnames):
    path = f'{parent_dir_path}/{"/".join(intermediate_dirs)}'

    for dirname in dirnames:
        dir_path = f'{path}/{dirname}'
        os.makedirs(dir_path, exist_ok=True)
    
    return

def _process_files(input_dir,output_dir, filenames):
    pattern = re.compile('[-\.]')
    for filename in filenames:

        speaker, chapter, some_id, ext = pattern.split(filename) # re.split(r'[-\.]', filename)

        src_path = f'{input_dir}/{speaker}/{chapter}/{filename}'
        
        if ext == 'txt':
            dst_path = f'{output_dir}/{speaker}/{chapter}/{filename}'
            shutil.copyfile(src_path, dst_path)
            continue

        dst_path = f'{output_dir}/{speaker}/{chapter}/{speaker}-{chapter}-{some_id}.wav'

        data, samplerate = sf.read(src_path)
        sf.write(dst_path, data, samplerate)

def convert_flac_to_wav(input_dir, output_dir):
    items = os.walk(input_dir)

    # First item contains the speakers directory names
    _, speakers, _ = items.__next__()

    _create_dirs_given_dirnames(output_dir, [''], speakers)

    for (path, dirnames, filenames) in items:
        # An item will contain either the dirname or list of files in a directory
        # And if dirname is non-empty list then its a list of chapters for a speaker
        # Else its the filenames list 
        if dirnames:
            speaker = path.split('/')[-1]
            _create_dirs_given_dirnames(output_dir, [speaker], dirnames)
            continue
        
        _process_files(input_dir, output_dir, filenames)

def _add_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("input directory", help="Directory containing LibriSpeech flac files.")
    parser.add_argument("output directory", help="Direcotry to save converted wav files.")
    
    args = parser.parse_args()
    return args.__dict__

if __name__ == "__main__":
    args = _add_argument_parser()

    input_dir = args['input directory']
    output_dir = args['output directory']

    convert_flac_to_wav(input_dir, output_dir)