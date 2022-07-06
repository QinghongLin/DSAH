from utils.metric import eval_map
import numpy as np

if __name__ == '__main__':
    map_1000, map_5000, map_all = eval_map('data/output/cifar-10/debug_0_epoch_0', dataset='cifar')
    print('map:{:.4f}, map_1000:{:.4f}, map_5000:{:.4f}'.format(map_all, map_1000, map_5000))
