import sys
sys.path.append('../audio2tokens')
from audio2tokens import *
download_dir=''

def retrieve_tokens(audiofile):
	audio_path_raw = _raw_audio(audiofile)
    tokens = _get_text(audio_path_raw)
	return tokens 

def rank(namefile):
	audiofile=os.path.join(download_dir,namefile)
    tokens=retrieve_tokens(audiofile)
	return {'score':0.1, 'rhymes_scheme':6}