#import SimpleHTTPServer
#import SocketServer
#!/usr/bin/env python
 
from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
import argparse
import cgi
import os
import time
import json
import argparse
from urlparse import urlparse, parse_qs
from Judge import rank
debug=True

class RapJudgeServer(BaseHTTPRequestHandler):  
  #handle GET command


	def do_GET(self):
		print("hey!")
		self.send_response(200)
		self.send_header('Content-type','text-html')
		self.send_header('Access-Control-Allow-Origin', '*')
		self.end_headers()

		o = urlparse(self.path)
		params=parse_qs(o.query)
		values=rank(params['namefile'][0])
		self.wfile.write(json.dumps(values))
	

	def do_POST(self):
		'''
		Handle POST requests.
		'''
		print('POST %s' % (self.path))

		ctype, pdict = cgi.parse_header(self.headers['content-type'])
		if ctype == 'multipart/form-data':
			postvars = cgi.parse_multipart(self.rfile, pdict)
		elif ctype == 'application/x-www-form-urlencoded':
			length = int(self.headers['content-length'])
			postvars = cgi.parse_qs(self.rfile.read(length), keep_blank_values=1)
		else:
			postvars = {}

        # Get the "Back" link.
        #back = self.path if self.path.find('?') < 0 else self.path[:self.path.find('?')]

        # Print out logging information about the path and args.
		print('TYPE %s' % (ctype))
		print('PATH %s' % (self.path))
		print('ARGS %d' % (len(postvars)))
		'''
		if len(postvars):
			i = 0
			for key in sorted(postvars):
				print('ARG[%d] %s=%s' % (i, key, postvars[key]))
				i += 1
		'''
		data=postvars['data']
		HARM_2016_PATH = os.environ["HARM_2016_PATH"]

		namefile=os.path.join(HARM_2016_PATH,'waves','%s.wav'%str(int(time.time()%14000000000*100000)))
		with open(namefile,'wb') as f:
			f.write(data[0])
		# Tell the browser everything is okay and that there is
		# HTML to display.
		self.send_response(200)  # OK
		self.send_header('Content-type', 'text/html')		
		self.send_header('Access-Control-Allow-Origin', '*')
		self.end_headers()
		values=rank(namefile)
		self.wfile.write(json.dumps(values))
		#self.wfile.write(json.dumps({'ciao':5}))


def run(port, address):
	print('http server is starting...')
	PORT=80
	server_address = (address, port)
	#server_address = ('10.22.12.169', PORT)
	httpd = HTTPServer(server_address, RapJudgeServer)	
	print('http server is running on address %s:%d'%(address,port))
  	httpd.serve_forever()
 

def build_argument_parser():
    parser = argparse.ArgumentParser(description=__doc__, add_help=True,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('-p', '--port', type=int, default=8000, action='store',
                        help='Port for request')
    parser.add_argument('-a', '--address', type=str, default="127.0.0.1", action='store',
                        help='Address from remote (default is local).')
    return parser



if __name__ == '__main__':
	parser = build_argument_parser()
	args = parser.parse_args()
	run(args.port, args.address)

	#Handler = SimpleHTTPServer.SimpleHTTPRequestHandler
	#httpd = SocketServer.TCPServer(("", PORT), Handler)
	#print "serving at port", PORT
	#httpd.serve_forever()