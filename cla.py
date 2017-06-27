import numpy as np 
import random as rd 

class resection(object):
    """docstring for resection"""
    def __init__(self, img_file, contr_file):
        super(resection, self).__init__()
        self.img_file = img_file
        self.contr_file = contr_file

        self.f = 4547.93519
        self.x0 = 47.48571
        self.y0 = 12.02756

        self._a1, self._a2, self._a3, \
        self._b1, self._b2, self._b3, \
        self._c1, self._c2, self._c3, \
        self._Xs, self._Ys, self._Zs = self.guess_params()

        self.img_poi = self.get_position(self.img_file)
        self.con_poi = self.get_position(self.contr_file)

        self.x, self.y, self.X, self.Y, self.Z = self.get_coords()

        self.img_h = 4008
        self.img_w = 5344

        

    def iteration_process(self):
        self.transform_coords()
        for i in range(100):
            self.iterator()
            print(self._Xs, self._Ys, self._Zs)

        print("Calculating o p k ...")
        w = np.arcsin(-self._b3)
        cosw = np.cos(w)

        sinphi = - self._a3 / cosw
        phi = np.arcsin(sinphi)

        cosk = self._b2 / cosw
        k = np.arccos(cosk)
        print("complete o p k ")
        ob = np.array((self._Xs, self._Ys, self._Zs,phi, w, k)).reshape((1, 6))
        np.savetxt("内方位元素.txt", ob, fmt='%.5f', delimiter = '  ' ,
            header='Xs         Ys         Zs         phi      omega    kappa')


    def transform_coords(self):
        # 像点坐标为扫描坐标，原点在像片的左上角，单位为pixel。
        # 应将坐标原点平移至像片中心，作为像平面坐标系（x轴向右、y轴向上），代入共线方程。

        self.x = self.x - self.img_w/2
        self.y = -self.y + self.img_h/2
        

    def iterator(self):
        # 对这个值进行迭代
        self.B, self.L = self.get_BL(self.x, self.y, self.X, self.Y, self.Z)
        self.requirments()
        self.calc_X()
        self.update_params()

    def requirments(self):
        '''在调用 calc_X() 之前需要进行的计算'''

        self.get_C()
        self.W = np.dot(self.B.T, self.L)
        self.get_Wx()
        self.get_Nbb()
        self.get_Ncc()

    def get_coords(self):
        # 获得 x,y, X,Y,Z 的值
        _, x, y = self.img_poi[0, :]
        _, X, Y, Z = self.con_poi[0, :]
        return x, y, X, Y, Z


    def get_Wx(self):
        # 计算/跟新 Wx 的值
        self.Wx = np.array([self._a1**2 + self._a2**2 + self._a3**2 - 1,
                            self._b1**2 + self._b2**2 + self._b3**2 - 1,
                            self._c1**2 + self._c2**2 + self._c3**2 - 1,
                            self._a1*self._a2 + self._b1*self._b2 + self._c1*self._c2,
                            self._a1*self._a3 + self._b1*self._b3 + self._c1*self._c3,
                            self._a2*self._a3 + self._b2*self._b3 + self._c2*self._c3])        

    def get_position(self, fname):
        '''从文件中读取数据(ndarray)'''
        f = open(fname)
        ob = f.readlines()
        ob = [l.split() for l in ob]
        m, n = np.array(ob).shape
        for i in range(m):
            for j in range(n):
                ob[i][j] = float(ob[i][j])

        return np.array(ob)

    def guess_params(self):
        t = [rd.random() for i in range(12)]
        # return [1.1, 1.3, 1.5, 1.1, 1, 1, 1, 1, 1.5, 1000, 1000, 1000]
        # return [1, 0, 0, 0, 1, 0, 0, 0, 1, 1000, 1000, 1000]
        return [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]

        return t

    def get_BL(self, x, y, X, Y, Z):
        ''' 对于一个点 (x,y) (X,Y,Z)计算/跟新 B L 的值 
        x:
        y:
        X:Y Z 每一个控制点的像方坐标和物方坐标
        
        '''
        # 对每一个所给的点 (X,Y,Z), '_ba'这几个参数都是不同的
        X_ba = self._a1*(X-self._Xs) + self._b1*(Y-self._Ys) + self._c1*(Z-self._Zs) 
        Y_ba = self._a2*(X-self._Xs) + self._b2*(Y-self._Ys) + self._c2*(Z-self._Zs)
        Z_ba = self._a3*(X-self._Xs) + self._b3*(Y-self._Ys) + self._c3*(Z-self._Zs)

        b11 = (self._a1*self.f + self._a3*(x-self.x0)) / Z_ba
        b12 = (self._b1*self.f + self._b3*(x-self.x0)) / Z_ba
        b13 = (self._c1*self.f + self._c3*(x-self.x0)) / Z_ba
        b14 = -self.f*(X-self._Xs) / Z_ba
        b15 = 0
        b16 = -(x-self.x0)*(X-self._Xs) / Z_ba
        b17 = -self.f * (Y - self._Ys) / Z_ba
        b18 = 0
        b19 = -(x-self.x0)*(Y-self._Ys) / Z_ba

        b1a = -self.f * (Z - self._Zs) / Z_ba
        b1b = 0
        b1c = -(x-self.x0) * (Z-self._Zs) / Z_ba

        b21 = (self._a2*self.f + self._a3*(y-self.y0)) / Z_ba
        b22 = (self._b2*self.f + self._b3*(y-self.y0)) / Z_ba
        b23 = (self._c2*self.f + self._c3*(y-self.y0)) / Z_ba
        b24 = 0

        b25 = -self.f*(X-self._Xs) / Z_ba
        b26 = -(y-self.y0)*(X-self._Xs) / Z_ba
        b27 = 0
        b28 = -self.f * (Y-self._Ys) / Z_ba
        b29 = -(y-self.y0) * (Y - self._Ys) / Z_ba

        b2a = 0
        b2b = -self.f*(Z-self._Zs) / Z_ba
        b2c = -(y-self.y0)*(Z-self._Zs) / Z_ba

        t = [[b11, b12, b13, b14, b15, b16, b17, b18, b19, b1a, b1b, b1c], 
                [b21, b22, b23, b24, b25, b26, b17, b28, b29, b2a, b2b, b2c]]
        
        l1 = (x-self.x0) + self.f * X_ba / Z_ba
        l2 = (y-self.y0) + self.f * Y_ba / Z_ba
        l = np.array((l1, l2))
        return np.array(t), l
    
    def get_C(self):
        '''每一组估计值对应一个C值'''
        t = [[0, 0, 0, 2*self._a1, 2*self._a2, 2*self._a3, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 2*self._b1, 2*self._b2, 2*self._b3, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 2*self._c1, 2*self._c2, 2*self._c3],
             [0, 0, 0, self._a2, self._a1, 0, self._b2, self._b1, 0, self._c2, self._c1, 0],
             [0, 0, 0, self._a3, 0, self._a1, self._b3, 0, self._b1, self._c3, 0, self._c1],
             [0, 0, 0, 0, self._a3, self._a2, 0, self._b3, self._b2, 0, self._c3, self._c2]]
        self.C = np.array(t)
    
    def get_Nbb(self, P=None):
        self.Nbb = np.dot(self.B.T, self.B)

    def get_Ncc(self):#C, Nbb
        t = np.linalg.inv(self.Nbb)
        t = np.dot(self.C, t)
        self.Ncc = np.dot(t, self.C.T)
    
    def calc_X(self):

        inv_Nbb = np.linalg.inv(self.Nbb)
        inv_Ncc = np.linalg.inv(self.Ncc)

        t1 = np.dot(inv_Nbb, self.C.T)
        t2 = np.dot(t1, inv_Ncc)
        t3 = np.dot(t2, self.C)
        t4 = np.dot(t3, inv_Nbb)
        t = inv_Nbb - t4

        t = np.dot(t, self.W)

        p2 = np.dot(inv_Nbb, self.C.T)
        p22 = np.dot(inv_Ncc, self.Wx)

        p2 = np.dot(p2, p22)

        self.dPabc = t - p2
        
    def givens_rotation(self, A):
        """Givens变换"""
        (r, c) = np.shape(A)
        Q = np.identity(r)
        R = np.copy(A)
        (rows, cols) = np.tril_indices(r, -1, c)
        for (row, col) in zip(rows, cols):
            if R[row, col] != 0:  # R[row, col]=0则c=1,s=0,R、Q不变
                r_ = np.hypot(R[col, col], R[row, col])  # d
                c = R[col, col]/r_
                s = -R[row, col]/r_
                G = np.identity(r)
                G[[col, row], [col, row]] = c
                G[row, col] = s
                G[col, row] = -s
                R = np.dot(G, R)  # R=G(n-1,n)*...*G(2n)*...*G(23,1n)*...*G(12)*A
                Q = np.dot(Q, G.T)  # Q=G(n-1,n).T*...*G(2n).T*...*G(23,1n).T*...*G(12).T
        return (Q, R)

    def update_params(self):

            self._Xs += self.dPabc[0]
            self._Ys += self.dPabc[1]
            self._Zs += self.dPabc[2]
            self._a1 += self.dPabc[3]
            self._a2 += self.dPabc[4]
            self._a3 += self.dPabc[5]
            self._b1 += self.dPabc[6]
            self._b2 += self.dPabc[7]
            self._b3 += self.dPabc[8]
            self._c1 += self.dPabc[9]
            self._c2 += self.dPabc[10]
            self._c3 += self.dPabc[11]

class all_resection(resection):
    """将所有的点一起进行平差，利用方向余弦的额后方交会"""
    def __init__(self, img_file, contr_file):
        super(all_resection, self).__init__(img_file, contr_file)


    def iterator(self):

        self.B, self.L = self.get_BL(self.x, self.y, self.X, self.Y, self.Z)
        self.requirments()
        self.calc_X()
        self.update_params()

    def get_coords(self):

        x, y = self.img_poi[:, 1], self.img_poi[:, 2]
        X, Y, Z = self.con_poi[:, 1], self.con_poi[:, 2], self.con_poi[:, 3]
        return x, y, X, Y, Z

    def get_BL(self, x, y, X, Y, Z):

        tB = np.zeros((1, 12))
        tL = np.zeros(( 1 ))
        m = len(x)
        for i in range(m):
            tb, tl = super().get_BL(x[i], y[i], X[i], Y[i], Z[i])
            
            tB = np.vstack((tB, tb))
            tL = np.concatenate((tL, tl))
        tB = tB[1:, :]
        tL = tL[1:]
        return tB, tL
        
class Tester(all_resection):
    """docstring for Tester"""
    def __init__(self, img_file , contr_file ,
                        x0=None, y0=None, f = None, img_shape:"Height,Width"=None):
        super(Tester, self).__init__( img_file, contr_file)
        
        if img_shape!=None:
            self.img_h, self.img_w = img_shape        
        if x0!=None:    self.x0 = x0
        if y0!=None:    self.y0 = y0
        if f != None:   self.f = f 

        
a = all_resection('data/像点坐标.txt', 'data/控制点坐标.txt')
a = Tester('data/像点坐标.txt', 'data/控制点坐标.txt', x0 = 47.48571, y0= 12.02756, f= 4547.93519, img_shape=(4008,5344))
# a = Tester('data/像点坐标.txt', 'data/控制点坐标.txt', x0 = 47.48571, y0= 12.02756, f= 4547.93519, img_shape=(400,534))
a.iteration_process()
# for i in range(100):
#     a.iterator()
#     print(a._Xs, a._Ys, a._Zs)



