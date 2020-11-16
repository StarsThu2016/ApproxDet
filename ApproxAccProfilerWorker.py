'''
The worker script to perform the profiling jobs.
Architecture: the worker sends two types of messages,
(1) Request:
(2) Done: [config]
The master responds with,
(1) Send: [config]
(2) None

Usage:
python3 ApproxAccProfilerWorker.py --ip=localhost --port=5050 \
  --dataset_prefix=/data1/group/mlgroup/train_data/ILSVRC2015/ --cpu=0
'''

import http.client
import pickle, argparse, subprocess

parser = argparse.ArgumentParser(description='Client-side main script.')
parser.add_argument('--ip', dest='ip', required=True, help='ip of the server.')
parser.add_argument('--port', dest='port', required=True, help='port of the server.')
parser.add_argument('--dataset_prefix', dest='dataset_prefix', required=True, 
  help='Dataset path.')
parser.add_argument('--cpu', dest='cpu', required=True, help='CPU core to take.')
args = parser.parse_args()

def request(send_dict): # (1) Ask for new (2) report the last finished one

    send_bytes = pickle.dumps(send_dict)
    connection = http.client.HTTPConnection("{}:{}".format(args.ip, args.port))
    connection.request('POST', '/', body = send_bytes, headers = {'Content-type': 'application/json'})
    response = connection.getresponse()
    content_length = response.getheader('Content-Length')
    receive_bytes = response.read(content_length)
    config = pickle.loads(receive_bytes)
    return config

def do(config):

    if not config:
        return False
    dataset, nprop, shape, si, tracker, ds = config
    det = 'test/VID_{}set_nprop{}_shape{}_det.txt'.format(dataset, nprop, shape)
    output = 'test/VID_{}set_nprop{}_shape{}.txt'.format(dataset, nprop, shape)
    cmd = 'taskset -c {} python3 ApproxTracking.py '.format(args.cpu) + \
          '--imagefiles=test/VID_{}img_full.txt '.format(dataset) + \
          '--dataset_prefix={} '.format(args.dataset_prefix) + \
          '--detection_file={} '.format(det) + \
          '--si={} --tracker_ds={}_ds{} '.format(si, tracker, ds) + \
          '--output={}'.format(output)
    print("Running {}".format(cmd))
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    stdout, stderr = process.communicate()
    return True

while True:
    config = request({"request": True})
    if do(config):
        config = request({"done": config})
    else:
        break
