#import SimpleHTTPServer
#import SocketServer
#!/usr/bin/env python
 
from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
import argparse
import cgi
import os
import json
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
		if len(postvars):
			i = 0
			for key in sorted(postvars):
				print('ARG[%d] %s=%s' % (i, key, postvars[key]))
				i += 1

		# Tell the browser everything is okay and that there is
		# HTML to display.
		self.send_response(200)  # OK
		self.send_header('Content-type', 'text/html')		
		self.send_header('Access-Control-Allow-Origin', '*')
		self.end_headers()

		self.wfile.write(json.dumps({'ciao':5}))







      
def run():
	print('http server is starting...')
	PORT=80
	server_address = ('127.0.0.1', 8000)
	#server_address = ('10.22.12.169', PORT)
	httpd = HTTPServer(server_address, RapJudgeServer)	
	print('http server is running...')
  	httpd.serve_forever()
  
if __name__ == '__main__':
	run()

	#Handler = SimpleHTTPServer.SimpleHTTPRequestHandler
	#httpd = SocketServer.TCPServer(("", PORT), Handler)
	#print "serving at port", PORT
	#httpd.serve_forever()