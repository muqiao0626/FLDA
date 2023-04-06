import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

def rifle(A, B, init, k, eta = 0.01, convergence = 0.001, maxiter = 5000):
    n, p = B.shape
    x = init
    x = init/np.linalg.norm(x)
    criteria = 1e10
    iter = 0
    while (criteria > convergence and iter <= maxiter):
        #print (iter)
        #print (criteria)
        rho = (x.T.dot(A).dot(x))/(x.T.dot(B).dot(x))
        C = np.identity(p) + eta/rho*(A-rho*B)
        xprime = C.dot(x)
        xprime = xprime/np.linalg.norm(xprime)
        # Truncation
        truncate_value = np.sort(abs(xprime), axis = 0)[-k]
        xprime[abs(xprime) < truncate_value] = 0
        xprime = xprime/np.linalg.norm(xprime)
        criteria = np.sqrt(sum((x-xprime)**2))
        x = xprime
        iter = iter + 1
    return xprime

# BEGIN 3d complete
class FLDA_3d_complete:
    def __init__(self, n_components = [1,1,1]):
        self.nc = n_components
        self.vs = None
        self.eigvs = None
        self.inits = None
        self.Ts = None
        self.D = None
        self.svs = None

    def fit(self, x, y, b = [1,1,1]): # x is in the format of cells x genes, y is in the format of cells x 3, labels of y start from 0
        nc1 = self.nc[0]
        nc2 = self.nc[1]
        nc3 = self.nc[2]
        n,p = x.shape

        yi = y[:, 0]
        li = max(yi)+1
        yj = y[:, 1]
        lj = max(yj)+1
        yk = y[:, 2]
        lk = max(yk)+1

        nijk = np.array([[[sum((yi == i) * (yj == j) * (yk == k)) for k in range(lk)] for j in range(lj)] for i in range(li)])
        xijk = np.array([[[x[(yi == i) * (yj == j) * (yk == k), :].mean(axis = 0) for k in range(lk)] for j in range(lj)] for i in range(li)])
        xi = xijk.mean(axis = 2).mean(axis = 1)
        xj = xijk.mean(axis = 2).mean(axis = 0)
        xk = xijk.mean(axis = 0).mean(axis = 0)
        x_bar = xi.mean(axis = 0)
        
        A = (xi-x_bar).T.dot(xi-x_bar)/(li-1)
        B = (xj-x_bar).T.dot(xj-x_bar)/(lj-1)
        C = (xk-x_bar).T.dot(xk-x_bar)/(lk-1)
        D = np.zeros((p,p))
        
        for i in range(li):
            for j in range(lj):
                for k in range(lk):  
                    d = x[(yi == i) * (yj == j) * (yk == k), :] - xi[i] - xj[j] - xk[k] + 2*x_bar
                    D = D + d.T.dot(d)/nijk[i,j,k]
        D = D/(n-li*lj-lk+2)

        T1 = b[0]*A-b[1]*B-b[2]*C
        T2 = b[1]*B-b[0]*A-b[2]*C
        T3 = b[2]*C-b[0]*A-b[1]*B
        
        s, u = np.linalg.eigh(D)
        s = np.diag(1/np.sqrt(s))
        z = u.dot(s)

        M1_ = z.T.dot(T1).dot(z)
        s1_, u1_ = np.linalg.eigh(M1_)
        v1s = z.dot(u1_[:,0-nc1:])

        M2_ = z.T.dot(T2).dot(z)
        s2_, u2_ = np.linalg.eigh(M2_)
        v2s = z.dot(u2_[:,0-nc2:])

        M3_ = z.T.dot(T3).dot(z)
        s3_, u3_ = np.linalg.eigh(M3_)
        v3s = z.dot(u3_[:,0-nc3:])

        self.vs = [v1s, v2s, v3s]

        self.inits = [v1s[:,0], v2s[:,0], v3s[:,0]]
        self.Ts = [T1, T2, T3]
        self.D = D

        self.eigvs = [s1_[0-nc1:], s2_[0-nc2:], s3_[0-nc3:]]
        
        return self.vs, self.eigvs

    def sparse_fit(self, x, y, b = [1,1,1], ks = [10,10,10], trace = False, eta = 0.01, Ts = None, D = None, inits = None):
        if (Ts is None) or (D is None):
            nc1 = self.nc[0]
            nc2 = self.nc[1]
            nc3 = self.nc[2]
            n,p = x.shape

            yi = y[:, 0]
            li = max(yi)+1
            yj = y[:, 1]
            lj = max(yj)+1
            yk = y[:, 2]
            lk = max(yk)+1

            nijk = np.array([[[sum((yi == i) * (yj == j) * (yk == k)) for k in range(lk)] for j in range(lj)] for i in range(li)])
            xijk = np.array([[[x[(yi == i) * (yj == j) * (yk == k), :].mean(axis = 0) for k in range(lk)] for j in range(lj)] for i in range(li)])
            xi = xijk.mean(axis = 2).mean(axis = 1)
            xj = xijk.mean(axis = 2).mean(axis = 0)
            xk = xijk.mean(axis = 0).mean(axis = 0)
            x_bar = xi.mean(axis = 0)
            
            A = (xi-x_bar).T.dot(xi-x_bar)/(li-1)
            B = (xj-x_bar).T.dot(xj-x_bar)/(lj-1)
            C = (xk-x_bar).T.dot(xk-x_bar)/(lk-1)
            D = np.zeros((p,p))
            
            for i in range(li):
                for j in range(lj):
                    for k in range(lk):  
                        d = x[(yi == i) * (yj == j) * (yk == k), :] - xi[i] - xj[j] - xk[k] + 2*x_bar
                        D = D + d.T.dot(d)/nijk[i,j,k]
            D = D/(n-li*lj-lk+2)

            T1 = b[0]*A-b[1]*B-b[2]*C
            T2 = b[1]*B-b[0]*A-b[2]*C
            T3 = b[2]*C-b[0]*A-b[1]*B
            Ts = [T1, T2, T3]

        if inits is None:
            inits = self.inits

        dd, ee = np.linalg.eigh(D)
        print ("Largestest eigenvalue:", dd[-1])
        print ("Eta:", eta)
        assert (eta*dd[-1] < 1)

        svs = []
        for i in range(3):
            svs.append(rifle(Ts[i], D, inits[i], ks[i], eta = eta, convergence = 0.001, maxiter = 5000))

        self.svs = svs
        return self.svs

    def transform(self, x):
        x1 = x.dot(self.vs[0])
        x2 = x.dot(self.vs[1])
        x3 = x.dot(self.vs[2])
        return x1, x2, x3

    def sparse_transform(self, x):
        sx1 = x.dot(self.svs[0])
        sx2 = x.dot(self.svs[1])
        sx3 = x.dot(self.svs[2])
        return sx1, sx2, sx3

# END 3d complete
