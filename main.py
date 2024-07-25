import os

from ucs_alg_node import AlgNode, AlgSubmitter, AlgNodeWeb, AlgTask, utils
from hir_crowd_count_alg import HirCrowdCountAlg
import time


task_id = 'task112'
node_id = 'node001'
out_topic = 'ucs/alg/result'


def main():
    cfg = {
        'id': node_id,
        'name': 'alg_name',
        'mode': 'batch',
        'max_task': 10,
        'model_dir': os.path.join(utils.get_cwd(), 'models'),  # could be file path or url or model name
        'alg_id': 'alg_id123', # only effective in batch mode
        'web_port':9996
    }

    task = AlgTask(id=task_id,
                   ts=utils.current_time_milli(),
                   sources= ['rtsp://localhost:9111/123',
                             'mqx://localhost:8011/1123'])

    alg_cfg = {
        # only effective in stream mode
        'id': 'alg_id123',
        # only effective in stream mode
        'model': 'model_v1.pth',  # could be file path or url or model name
    }

    out_cfg = {

        'dest': '62.234.16.239:1883',
        #TODO: remove the addr
        'mode': 'mqtt',
        'username': 'admin',
        'passwd': 'admin1234',
        'topic': out_topic
    }

    alg = HirCrowdCountAlg(alg_cfg['id'], alg_cfg['model'])

    submitter = AlgSubmitter(
        dest=out_cfg['dest'],
        mode=out_cfg['mode'],
        username=out_cfg['username'],
        passwd=out_cfg['passwd'],
        id=cfg['id'],
        topic=out_cfg['topic']  # if in db mode, can be omitted
    )

    node_cfg = {
        'id': cfg['id'],
        'name': cfg['name'],
        'model_dir': cfg['model_dir'],  # could be file path or url or model name
        'max_task': cfg['max_task'], # only effective in batch mode
        'mode': cfg['mode'],
        'task': task,
        'alg': alg,
        'out': submitter
    }

    node = AlgNode(max_task=10, cfg=node_cfg, task=task)
    node_web_api = AlgNodeWeb(cfg['web_port'], node)

    node.start()
    node_web_api.run()

    print('start node')
    while True:
        time.sleep(5)
        # node.stop()
        # print('stop node, exit')
        # break

if __name__ == '__main__':
    main()
