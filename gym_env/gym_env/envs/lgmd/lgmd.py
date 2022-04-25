import numpy as np
from scipy import signal
import cv2
import time

class LGMD():
    def __init__(self, type="origin"):
        
        self.type = type
        print(self.type)
        
        self.init_ok = False
        self.img_g_curr = None
        self.img_g_prev = None
        
        self.p_layer = None  # P = I - I'
        self.p_prev = None   # previous p layer
        self.i_layer = None  # i = I' * wi
        self.s_layer = None  # S = P - i * Wi
        
        self.wi = np.array([[0.125, 0.25, 0.125],
                            [0.25, 0, 0.25],
                            [0.125, 0.25, 0.125]])
        
        self.Ki = 0.35
        
        if self.type == 'norm':
            self.N_cell = 6e4
        else:
            self.N_cell = 1e6
        self.lgmd_out = None
        
        self.split_row_num = 1
        self.split_col_num = 10
        self.split_num_total = self.split_row_num * self.split_col_num
        self.out_split = np.zeros(self.split_num_total)
        self.out_split_old = np.zeros(self.split_num_total)
        self.out_split_expand_rate = np.zeros(self.split_num_total)
        
        self.run_time = 0
        
    def update(self, img_g):
        
        self.img_g_curr = img_g
        
        if self.type == 'norm':
            # preprocess the input gray image use image moment (similar to edge detection)
            start = time.time()
            self.img_g_curr = self.get_moment_norm(img_g)
            end = time.time()
            
            self.run_time += (end-start)
            
        if self.type == 'edge':
            # use edge detection directly to get s-layer output
            start = time.time()
            self.img_g_edge = self.get_edge(img_g)  # edge = 255
            end = time.time()
            self.run_time += (end-start)
            self.s_layer = self.img_g_edge / 255      # edge = 1
            self.lgmd_out = np.sum(abs(self.s_layer))
            
            s_layer_split = self.split_array(abs(self.s_layer), self.split_row_num, self.split_col_num)
            
            for i, img in enumerate(s_layer_split):
                    self.out_split[i] = np.mean(img)
                    
                    if self.out_split_old[i] == 0:
                        self.out_split_expand_rate[i] = 0
                    else:
                        self.out_split_expand_rate[i] = (self.out_split[i] - self.out_split_old[i]) / self.out_split_old[i]
                        if self.out_split_expand_rate[i] < 0:
                            self.out_split_expand_rate[i] = 0
                    
                    self.out_split_old[i] = self.out_split[i]
        else:
            if self.init_ok:
                    # get p i s layer output 
                self.p_layer = self.img_g_curr - self.img_g_prev
                self.i_layer = signal.convolve2d(self.p_prev, self.wi, boundary='symm',mode='same')
                self.s_layer = self.p_layer - self.i_layer * self.Ki
                
                s_layer_sum = np.sum(abs(self.s_layer))
                
                if self.type == 'norm': 
                    Kf = -s_layer_sum/self.N_cell
                    a = np.exp(Kf)
                    self.lgmd_out = 1/(1+a)
                else:
                    Kf = -(s_layer_sum - 6e6)/self.N_cell
                    a = np.exp(Kf)
                    self.lgmd_out = 1/(1+a)
                    # self.lgmd_out = s_layer_sum
                
                s_layer_split = self.split_array(abs(self.s_layer) , self.split_row_num, self.split_col_num)
                
                # get out split and expand rate
                for i, img in enumerate(s_layer_split):
                    self.out_split[i] = np.sum(img)
                    
                    if self.out_split_old[i] == 0:
                        self.out_split_expand_rate[i] = 0
                    else:
                        self.out_split_expand_rate[i] = (self.out_split[i] - self.out_split_old[i]) / self.out_split_old[i]
                        if self.out_split_expand_rate[i] < 0:
                            self.out_split_expand_rate[i] = 0
                    
                    self.out_split_old[i] = self.out_split[i]
                
                
                # update previous data
                self.img_g_prev = self.img_g_curr
                self.p_prev = self.p_layer
            else:
                # init img_g_prev and p_prev
                self.img_g_prev = self.img_g_curr
                self.p_layer = self.img_g_curr - self.img_g_prev
                self.p_prev = self.p_layer
                self.i_layer = self.p_layer
                self.s_layer = self.p_layer
                self.lgmd_out = 0
                self.init_ok = True
            
    def get_moment_norm(self, img):

        w01 = np.array([[-1, 0, 1],
                    [-1, 0, 1],
                    [-1, 0, 1]])

        w10 = np.array([[1, 1, 1],
                    [0, 0, 0],
                    [-1, -1, -1]])

        w00 = np.ones((3,3)) / 9

        m01 = signal.convolve2d(img, w01, boundary='symm', mode='same')
        m10 = signal.convolve2d(img, w10, boundary='symm', mode='same')
        m00 = signal.convolve2d(img, w00, boundary='symm', mode='same')

        m_total = m01+m10
        with np.errstate(divide='ignore'):
            m_norm = m_total/m00
        
        # deal with nan
        m_norm[np.isnan(m_norm)] = 0
        
        return m_norm
    
    def get_edge(self, img):
        edge = cv2.Canny(img, 50, 150)
        return edge
    
    def split_array(self, arr, row, col):
        l = np.array_split(arr,row,axis=0)

        new_l = []
        for a in l:
            l = np.array_split(a,col,axis=1)
            test = l[0]
            new_l += l
        
        return new_l