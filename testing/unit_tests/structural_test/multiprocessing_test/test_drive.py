import os
import time
import subprocess


import socket
from contextlib import closing

def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

# TODO: note, for now carla and carla test are in the same GPU


def execute(gpu, exp_alias, city_name):
    # We automatically define which one is the

    print("Running ", __file__, " On GPU ",gpu, "of experiment name ", exp_alias)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu


    #vglrun - d:7.$GPU $CARLA_PATH / CarlaUE4 / Binaries / Linux / CarlaUE4 / Game / Maps /$TOWN - windowed - benchmark - fps = 10 - world - port =$PORT;
    #sleep    100000


    port = find_free_port()
    carla_path = os.environ['CARLA_PATH']


    subprocess.call(['DISPLAY=:$DISPLAY_NUMBER', 'vglrun -d:7.' + str(gpu),
                     carla_path + '/CarlaUE4/Binaries/Linux/CarlaUE4/' + city_name,
                     '-benchmark', '-fps=10', '-world-port='+str(port)])


    time.sleep(10)

