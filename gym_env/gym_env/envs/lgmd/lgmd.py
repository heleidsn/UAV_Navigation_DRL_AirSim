import numpy as np
from scipy import signal

class LGMD():
    def __init__(self, type="origin"):
        
        self.type = type      # type: origin, norm, edge
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
        
        self.lgmd_out = None
        
        self.img_lgmd = None
        
    def update(self, img_g):
        self.img_g_curr = img_g
        if self.type == 'norm':
            self.img_g_curr = self.get_moment_norm(img_g)
        
        if self.init_ok:
             # get p i s layer output 
            self.p_layer = self.img_g_curr - self.img_g_prev
            self.i_layer = signal.convolve2d(self.p_prev, self.wi, boundary='symm',mode='same')
            self.s_layer = self.p_layer - self.i_layer * self.Ki
            
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
        m_norm = m_total/m00
        
        # deal with nan
        m_norm[np.isnan(m_norm)]=0
        
        # norm to 0-255
        
        
        return m_norm
    
    def get_edge(self, img):
        return img