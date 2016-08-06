import os, sys, subprocess
import string

#HARM_2016_PATH = os.environ["HARM_2016_PATH"]
HARM_2016_PATH='/media/michele/Data/Repo/HAMR_2016/'
def _raw_audio(audio_path):

    audio_name = '.'.join(audio_path.split('.')[:-1])
    audio_extension = audio_path.split('.')[-1]

    audio_path_16k = audio_name+'_16k.'+audio_extension
    if os.path.exists(audio_path_16k):
        os.remove(audio_path_16k)
    audio_path_raw = audio_name+'.raw'

    if os.path.exists(audio_path_raw):
        os.remove(audio_path_raw)

    sox_call = 'sox %s  -c 1 -r 16000 %s' % (audio_path,audio_path_16k)
    os.system(sox_call)
    print(os.path.exists(audio_path_16k))
    
    ffmpeg_call = 'ffmpeg -i %s -f s16le -acodec pcm_s16le %s' % (audio_path_16k, audio_path_raw)
    os.system(ffmpeg_call)

    print '\n\traw audio writting at \n\t\t%s' % audio_path_raw
    print(os.path.exists(audio_path_raw))
    
    return audio_path_raw

def _get_text(audio_path_raw):
    os.system('export GOOGLE_APPLICATION_CREDENTIALS="/media/michele/Data/Repo/HAMR_2016/audio2tokens/google_api/a73d2853efab.json"')
    os.chdir(os.path.join(HARM_2016_PATH,'audio2tokens','nodejs-docs-samples','speech'))

    command = 'nodejs recognize %s' % audio_path_raw
    proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    out = proc.communicate()[0]
    print(out)
    out_lines = map(string.strip,out.split('\n') )
    transcript = [ l for l in out_lines if l.startswith('"transcript"')][0]
    t_tokens = transcript[1:-1].split(':')[1].strip('"').split()

    tokens = []
    for token in t_tokens:
        if token.startswith('"'):
            token = token[1:]
        if token.endswith('"'):
            token = token[:-1]
        tokens.append(token)

    return tokens
raw_audio=_raw_audio
get_text=_get_text
if __name__=="__main__":
    namefile='/home/michele/Downloads/cleaning.wav'
    audio_path =  os.path.abspath(namefile)#sys.argv[1])

    audio_path_raw = _raw_audio(audio_path)

    tokens = _get_text(audio_path_raw)

    print " tokens found : "
    print tokens




