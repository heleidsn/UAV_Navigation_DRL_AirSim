import numpy as np
import math
import time
import cv2
from scipy import signal


class LGMD():
    def __init__(self, type="norm"):

        self.type = type
        # print(self.type)

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

        self.Ki = 0.5

        self.N_cell = 76800  # 320*240

        # 根据N_cell，a_cell，激活阈值等来确定一个增益系数
        # 假设10000cell激活 触发阈值0.9
        threshold = 0.9
        a_cell = 10000 
        self.K_activate = -math.log((1-threshold)/threshold) * self.N_cell / a_cell
        # print(self.K_activate)

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
                self.i_layer = signal.convolve2d(self.p_prev, self.wi,
                                                 boundary='symm', mode='same')
                self.s_layer = self.p_layer - self.i_layer * self.Ki

                self.p_layer = self.safe_uint8(abs(self.img_g_curr.astype(np.int16) - self.img_g_prev.astype(np.int16)))
                self.i_layer = self.safe_uint8(abs(signal.convolve2d(self.p_prev, self.wi, boundary='symm', mode='same')))
                self.s_layer = self.safe_uint8(self.p_layer.astype(np.int16) - self.i_layer.astype(np.int16) * self.Ki)

                s_layer_sum = np.sum(abs(self.s_layer))
                s_layer_threshold = (self.s_layer > 50)
                s_layer_threshold_img = s_layer_threshold * 255
                self.s_layer_activated = s_layer_activated = s_layer_threshold_img
                s_layer_activated_sum = s_layer_activated.sum()

                Kf = -(s_layer_activated_sum * self.K_activate) / self.N_cell  # 0 - 1 
                a = np.exp(Kf)
                self.lgmd_out = 1 / (1 + a)

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
                self.p_layer = abs(self.img_g_curr - self.img_g_prev).astype(np.uint8)
                self.p_prev = self.p_layer
                self.i_layer = self.p_layer
                self.s_layer = self.p_layer
                self.s_layer_activated = self.p_layer
                self.lgmd_out = 0.5
                self.init_ok = True

    def get_moment_norm(self, img):

        w01 = np.array([[-1, 0, 1],
                    [-1, 0, 1],
                    [-1, 0, 1]])

        w10 = np.array([[1, 1, 1],
                    [0, 0, 0],
                    [-1, -1, -1]])

        w00 = np.ones((3,3)) / 9

        edge_coef = 0.15
        m01 = abs(signal.convolve2d(img, w01, boundary='symm', mode='same')) * edge_coef   # float64
        m10 = abs(signal.convolve2d(img, w10, boundary='symm', mode='same')) * edge_coef   # float64
        m00 = signal.convolve2d(img, w00, boundary='symm', mode='same')

        self.m01 = self.safe_uint8(m01)
        self.m10 = self.safe_uint8(m10)
        self.m00 = self.safe_uint8(m00)

        self.edge = edge = m01+m10
        with np.errstate(divide='ignore'):
            m_norm = edge/m00

        # deal with nan
        m_norm[np.isnan(m_norm)] = 0
        m_norm[m00==0] = 0
        # print(np.max(m_norm))
        m_norm = self.safe_uint8(m_norm*255) 

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

    def safe_uint8(self, array):
        # 在进行uint8变换之前，对数组进行限制，否则会出现问题 256变成0
        array_clip = np.clip(array, 0, 255)
        return array_clip.astype(np.uint8)
