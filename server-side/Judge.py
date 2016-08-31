import sys
sys.path.append('../audio2tokens')
from audio2tokens import *
download_dir='/home/michele/Downloads'
from rhyme import  get_score
import time


def remove_audiofiles(audio_path_raw):
    audio_path_16k = audio_path_raw.replace('.raw','_16k.wav')
    audio_path = audio_path_raw.replace('.raw','.wav')
    os.remove(audio_path)
    os.remove(audio_path_16k)    
    os.remove(audio_path_raw)


def retrieve_tokens(audiofile):
	audio_path_raw = raw_audio(audiofile.replace(':','-'))
	tokens = get_text(audio_path_raw)
	remove_audiofiles(audio_path_raw)
	print(tokens)
	return tokens 

def rank(namefile):
	#audiofile=os.path.join(download_dir,namefile)
	tokens=retrieve_tokens(namefile)

	if tokens :
		score = get_score({'text':tokens})
	else:
		score = None
		print "No tokens found"

	print(score)
	return {'score':score, 'rhymes_scheme':6}
