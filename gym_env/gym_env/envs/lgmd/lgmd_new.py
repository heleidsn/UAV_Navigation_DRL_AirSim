# 最新的lgmd算法 和openMV中的算法保持一致
import numpy as np
from scipy import signal
import cv2
import time

class LGMD():
    """
    Latest LGMD algorithm
    update: 
        get LGMD output using moment image
    get_image_moment:
        get moment image from gray image
    safe_uint8:
        在进行uint8变换之前，对数组进行限制，否则会出现问题 256变成0
    """
    def __init__(self):
        self.init_ok = False
        self.image_m_curr = None
        self.image_m_prev = None
        
        self.p_layer = None  # P = I - I'
        self.p_prev = None   # previous p layer
        self.i_layer = None  # i = I' * wi
        self.s_layer = None  # S = P - i * Wi
        
        self.w_i_old = np.array([[0.125, 0.25, 0.125],
                            [0.25, 0, 0.25],
                            [0.125, 0.25, 0.125]])
        
        self.w_i = np.ones(shape=(3,3))/9  # I层的扩散函数
        self.W_i = 2                    # I层抑制系数
        
        
    def update(self, image_moment):
        self.image_m_curr = image_moment
        if self.init_ok:
            self.p_layer = self.safe_uint8(abs(self.image_m_curr.astype(np.int16) - self.image_m_prev.astype(np.int16)))
            self.i_layer = self.safe_uint8(abs(signal.convolve2d(self.p_prev, self.w_i, boundary='symm', mode='same')))
            self.s_layer = self.safe_uint8(self.p_layer.astype(np.int16) - self.i_layer.astype(np.int16) * self.W_i)

            self.image_m_prev_vis = self.image_m_prev.copy()
            self.image_m_prev = self.image_m_curr
            self.p_prev_vis = self.p_prev.copy()
            self.p_prev = self.p_layer
        else:
            # 初始化
            self.image_m_prev = self.image_m_curr
            self.image_m_prev_vis = self.image_m_prev.copy()
            self.p_layer = abs(self.image_m_curr - self.image_m_prev).astype(np.uint8)
            self.p_prev = self.p_layer
            self.p_prev_vis = self.p_prev.copy()
            self.i_layer = self.p_layer
            self.s_layer = self.p_layer
            
            self.init_ok = True
    
    def get_image_moment(self, img_gray):
        """_summary_

        Args:
            img_gray (uint8 array): gray image

        Returns:
            _type_: _description_
        """
        w01 = np.array([[-1, 0, 1],
                    [-1, 0, 1],
                    [-1, 0, 1]])

        w10 = np.array([[1, 1, 1],
                    [0, 0, 0],
                    [-1, -1, -1]])
        
        w00 = np.ones((3,3)) / 9
        
        edge_coef = 0.15
        # 获取
        m01 = abs(signal.convolve2d(img_gray, w01, boundary='symm', mode='same')) * edge_coef   # float64
        m10 = abs(signal.convolve2d(img_gray, w10, boundary='symm', mode='same')) * edge_coef   # float64
        edge = m01 + m10
        m00 = signal.convolve2d(img_gray, w00, boundary='symm', mode='same')
        
        with np.errstate(divide='ignore'):
            m_norm = edge / m00
        
        m_norm[np.isnan(m_norm)] = 0
        m_norm[m00==0] = 0
        # print(np.max(m_norm))
        m_norm = self.safe_uint8(m_norm*255)  # 在两个图片相除之后，需要乘以255以恢复成可视的图片，否则范围0-255 
        
        vis_list = [img_gray, self.safe_uint8(m00), self.safe_uint8(m01), self.safe_uint8(m10), self.safe_uint8(edge), m_norm]
        
        return m_norm, vis_list
    
    def safe_uint8(self, array):
        # 在进行uint8变换之前，对数组进行限制，否则会出现问题 256变成0
        array_clip = np.clip(array, 0, 255)
        return array_clip.astype(np.uint8)