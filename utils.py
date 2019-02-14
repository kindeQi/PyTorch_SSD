import time
import datetime
import visdom
import numpy as np

class draw_scatter():
    def __init__(self, port=8889):
        self.scatter = None
        self.vis = visdom.Visdom(port=port)
    
    def __call__(self, idx_array, loss_array):
        '''
        idx_array: int
        loss_array: int
        '''
        tmp = np.column_stack((np.array(idx_array), np.array(loss_array)))
        if self.scatter == None:
            self.scatter = self.vis.line(np.array([loss_array]), np.array([idx_array]))
        else:
            self.scatter = self.vis.line(np.array([loss_array]), np.array([idx_array]), win=self.scatter, update='append')
            
class my_timer(object):
    def __init__(self):
        self.start_time = time.time()
        
    def __call__(self):
        return str(datetime.timedelta(seconds=int(time.time() - self.start_time)))

class mAP():
    def __init__(self):

    def __call__(self, )

if __name__ == "__main__":
    scatter = draw_scatter()
    prev_loss = 0
    for i in range(1000):
        idx = i

        loss = np.random.randint(i - 200, i + 200)
        # loss = prev_loss * 0.9 + 0.1 * loss

        scatter(idx, loss)