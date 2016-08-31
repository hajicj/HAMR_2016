import sys
sys.path.append('../audio2tokens')
from audio2tokens import *
download_dir='/home/michele/Downloads'
from rhyme import  get_score
import time


def remove_audiofiles(audio_path):
    audio_path_16k = audio_path.replace('.wav','_16k.wav')
    audio_path_raw = audio_path.replace('.wav','.raw')

    os.remove(audio_path)
    os.remove(audio_path_16k)    
    os.remove(audio_path_raw)


def retrieve_tokens(audiofile):
    #time.sleep(3)
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