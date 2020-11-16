'''
A master script to manage the profiling jobs.
Architecture: the worker sends two types of messages,
(1) Request:
(2) Done: [config]
The master responds with,
(1) Send: [config]
(2) None

Usage:
python3 ApproxAccProfilerMaster.py --port=5050 --logs=test/VID_tmp.log
'''

import os, random, argparse, pickle
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

parser = argparse.ArgumentParser(description='Master-side main script.')
parser.add_argument('--port', dest='port', required=True, help='Port.')
parser.add_argument('--logs', dest='logs', required=True, help='Log file.')
args = parser.parse_args()

ListPending, ExpDoing, ListDone = [], {}, []
fout = open(args.logs, "w")

class S(BaseHTTPRequestHandler):

    def _set_response(self, bytes):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(bytes)

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        receive_bytes = self.rfile.read(content_length)
        receive_dict = pickle.loads(receive_bytes)

        if "done" in receive_dict:
            item = receive_dict["done"]
            if str(item) in ExpDoing:
                del ExpDoing[str(item)]
                ListDone.append(item)
                print("Done: [{}], done={}!".format(item, len(ListDone)))
                print("Done: [{}], done={}!".format(item, len(ListDone)), file = fout)
            else:
                print("Error: got done msg of {} but cannot find!".format(item))
                print("Error: got done msg of {} but cannot find!".format(item),
                      file = fout)

        if "request" in receive_dict:
            if ListPending:
                # item is [dataset, nprop, shape, si, tracker, ds]
                item = ListPending.pop(0)
                ExpDoing[str(item)] = 1
                send_bytes = pickle.dumps(item)
                N = len(ListDone) + len(ExpDoing)
                print("Sent: [{}], done+doing={}!".format(item, N))
                print("Sent: [{}], done+doing={}!".format(item, N), file = fout)
                self._set_response(send_bytes)
            else:
                print("Nothing in ListPending, return empty dict!")
                print("Nothing in ListPending, return empty dict!", file = fout)
                self._set_response(pickle.dumps({}))
        else:
            self._set_response(pickle.dumps({}))

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):

    """handle requests in a separate thread."""

def run(handler_class=S, port=0):

    server_address = ('', port)
    httpd = ThreadedHTTPServer(server_address, handler_class)
    print("Server started!")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()

# 2*7*4*6*6 = 2016 experiments, val exp first, slow exp first
for dataset in ["val", "test"]:
    for nprop in [100, 50, 20, 10, 5, 3, 1]:
        for shape in reversed([224, 320, 448, 576]):
            for si in reversed([2, 4, 8, 20, 50, 100]):
                for tracker, ds in [("medianflow", 4), ("medianflow", 2),
                                    ("medianflow", 1), ("kcf", 4), 
                                    ("csrt", 4), ("bboxmedianfixed", 4)]:
                    output_file = "test/VID_{}set_".format(dataset) + \
                                  "nprop{}_shape{}_".format(nprop, shape) + \
                                  "si{}_{}_ds{}_det.txt".format(si, tracker, ds)
                    if os.path.exists(output_file):
                        ListDone.append([dataset, nprop, shape, si, tracker, ds])
                    else:
                        ListPending.append([dataset, nprop, shape, si, tracker, ds])

print("Number of experiments that we have some logs: {}".format(len(ListDone)))
print("Number of experiments that we have not touched: {}".format(len(ListPending)))
print("Number of experiments in total: {}".format(len(ListDone) + len(ListPending)))

run(port=int(args.port))
fout.close()
