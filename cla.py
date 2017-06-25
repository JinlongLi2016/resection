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

        
    def iterator(self):
        x, y, X, Y, Z = self.get_coords()
        self.x, self.y, self.X, self.Y, self.Z = x, y, X, Y, Z

        self.B, self.L = self.get_BL(x, y, X, Y, Z)
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
        _, x, y = self.img_poi[0, :]
        _, X, Y, Z = self.con_poi[0, :]
        return x, y, X, Y, Z


    def get_Wx(self):
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
        '''
        x:
        y:
        X:Y Z 每一个控制点的像方坐标和物方坐标
        
        '''
        # 对每一个所给的点 (X,Y,Z), '_ba'这几个参数都是不同的
        X_ba = self._a1*(X-self._Xs  ) + self._b1*(Y-self._Ys) + self._c1*(Z-self._Zs) 
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

        self.X = t - p2
    
    def update_params(self):
            # params = [self._Xs, self._Ys, self._Zs,
            #           self._a1, self._a2, self._a3,
            #           self._b1, self._b2, self._b3, 
            #           self._c1, self._c2, self._c3]
            # i = 0
            # for pms in params:
            #     pms += self.X[i]
            #     i += 1
            self._Xs += self.X[0]
            self._Ys += self.X[1]
            self._Zs += self.X[2]
            self._a1 += self.X[3]
            self._a2 += self.X[4]
            self._a3 += self.X[5]
            self._b1 += self.X[6]
            self._b2 += self.X[7]
            self._b3 += self.X[8]
            self._c1 += self.X[9]
            self._c2 += self.X[10]
            self._c3 += self.X[11]

class all_resection(resection):
    """docstring for all_resection"""
    def __init__(self, img_file, contr_file):
        super(all_resection, self).__init__(img_file, contr_file)


    def iterator(self):
        x, y, X, Y, Z = self.get_coords()
        self.x, self.y, self.X, self.Y, self.Z = x, y, X, Y, Z

        self.B, self.L = self.get_BL(x, y, X, Y, Z)
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
        
a = all_resection('data/像点坐标.txt', 'data/控制点坐标.txt')
print(a._Xs, a._Ys, a._Zs)
for i in range(100):
    a.iterator()
    print(a._Xs, a._Ys, a._Zs)
print(a._a1)

