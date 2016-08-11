import sys
sys.path.append('../audio2tokens')
from audio2tokens import *
download_dir='/home/michele/Downloads'
from rhyme import  get_score
import time
def retrieve_tokens(audiofile):
	#time.sleep(3)
	audio_path_raw = raw_audio(audiofile.replace(':','-'))
	tokens = get_text(audio_path_raw)
	print(tokens)
	return tokens 

def rank(namefile):
	#audiofile=os.path.join(download_dir,namefile)
	tokens=retrieve_tokens(namefile)
	score= get_score({'text':tokens})
	
	print(score)
	return {'score':score, 'rhymes_scheme':6}