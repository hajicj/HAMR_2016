download_dir=''

def rank(namefile):
	audiofile=os.path.join(download_dir,namefile)
	return {'score':0.1, 'rhymes_scheme':6}