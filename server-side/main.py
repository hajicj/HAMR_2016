#import SimpleHTTPServer
#import SocketServer
#!/usr/bin/env python
 
from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
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
		if debug:
			print(params)
			self.wfile.write(json.dumps(5))
		else:
			pass		
			values=rank(params['namefile'][0])
			self.wfile.write(json.dumps(values))
		return
      
def run():
	print('http server is starting...')
	PORT=80
	server_address = ('127.0.0.1', PORT)
	server_address = ('10.22.12.169', PORT)
	httpd = HTTPServer(server_address, RapJudgeServer)	
	print('http server is running...')
  	httpd.serve_forever()
  
if __name__ == '__main__':
	run()

	#Handler = SimpleHTTPServer.SimpleHTTPRequestHandler
	#httpd = SocketServer.TCPServer(("", PORT), Handler)
	#print "serving at port", PORT
	#httpd.serve_forever()