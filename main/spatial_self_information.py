import numpy as np
from scipy.stats import norm

class Surprisal:
    def __init__(self):
        pass
class AnalyticalSurprisal(Surprisal):
    def __init__(self):
        super().__init__()

    def correct(self):
        beta1 = 1 - 1 / self.d
        Mu_corrected = 0.
        for k in self.mean_dict.keys():
            if k[0] == k[1]:
                beta2 = 1 - 2 / self.mean_dict[k]
                Mu_corrected += beta1 * self.mean_coef_dict[k] * self.mean_dict[k] / beta2
            else:
                Mu_corrected += beta1 * self.mean_coef_dict[k] * self.mean_dict[k]

        return Mu_corrected

    def fit(self, cs, ns, w_map, ignores):
        self.d = w_map.shape[0]
        self.N, self.W = np.sum(ns), np.sum(w_map)
        self.xmean = np.sum(cs * ns) / np.sum(ns)
        self.xvar = np.sum((ns * (cs - self.xmean)) ** 2)

        self.mean, self.mean_dict, self.mean_coef_dict = self.compute_mean(cs, ns)
        self.mean_corrected = self.correct()
        self.std, self.std_dict, self.std_coef_dict = self.compute_std(cs, ns, ignores)
        self.scaling_factor = self.N / (self.W * self.xvar)

    def get_fitted_params(self):
        return self.mean_corrected, self.std, self.scaling_factor

    def get_moran_I_upper(self, Z, w_map):

        X = Z.flatten().reshape((1, -1)) - np.mean(Z)
        Y = np.matmul(w_map, X.T)

        scov = np.matmul(X, Y).flatten()

        return scov

    def get_moran_I(self, Z, w_map):

        X = Z.flatten().reshape((1, -1)) - np.mean(Z)
        Y = np.matmul(w_map, X.T)

        scov = np.matmul(X, Y).flatten()

        return scov * self.scaling_factor

    def get_probability(self, Z, w_map):
        moran_I_upper = self.get_moran_I_upper(Z, w_map)

        return norm.pdf(moran_I_upper, loc=self.mean, scale=self.std)

    def compute_wiki_S(self, w_map, Z):
        d = Z.shape[0]
        N = d * d
        W = np.sum(w_map)
        x = Z.flatten()
        xmean = np.mean(Z)
        S1, S2, S3, S4, S5 = 0., 0., 0., 0., 0.
        # for i in range(N):
        #     S2_tmp = 0.
        #     for j in range(N):
        #         S1 += (w_map[i,j] + w_map[j,i])**2 / 2
        #         S2_tmp += (np.sum(w_map[i,:]) + np.sum(w_map[:,i]))**2
        #
        #     S2 += S2_tmp
        S1 = 2 * np.sum(w_map**2)
        S2 = 4 * np.sum(np.sum(w_map, axis=1)**2)
        S3 = N * np.sum(np.power((x - xmean), 4)) / (np.sum(np.power(x - xmean, 2)))**2

        S4 = (N**2 - 3*N + 3) * S1 - N * S2 + 3*W**2
        S5 = (N**2 - N) * S1 - 2 * N * S2 + 6*W**2

        return S1, S2, S3, S4, S5, W

    def compute_wiki_mean_and_std(self, w_map, Z):
        N = Z.shape[0] * Z.shape[1]
        S1, S2, S3, S4, S5, W = self.compute_wiki_S(w_map, Z)
        Mu_wiki = -1 / (N - 1)
        Sigma_wiki = np.sqrt((N * S4 - S3 * S5) / ((N - 1) * (N - 2) * (N - 3) * W**2) - Mu_wiki**2)

        return Mu_wiki, Sigma_wiki

    @staticmethod
    def compute_mean(cs, ns):
        N = np.sum(ns)
        xmean = np.sum(cs * ns) / N
        mu = 0.

        mu_dict = {}
        mu_coef_dict = {}

        for i, (ci, ni) in enumerate(zip(cs, ns)):
            for j, (cj, nj) in enumerate(zip(cs, ns)):
                mu_coef = (ci - xmean) * (cj - xmean)
                if ci != cj:
                    mu_num = min(ni, nj) * 4 * max(ni, nj) / N
                    mu += mu_coef * mu_num
                else:
                    mu_num = ((ni - 1) * 4 * ni / N) - 1
                    mu += mu_coef * mu_num

                mu_dict[(i, j)] = mu_num
                mu_coef_dict[(i, j)] = mu_coef

        return mu, mu_dict, mu_coef_dict

    @staticmethod
    def compute_std(cs, ns, ignores=None):
        N = np.sum(ns)
        r_max = np.where(ignores == 0)[0][0]
        xmean = np.sum(cs * ns) / N
        var = 0.
        std_dict, std_coef_dict = {}, {}

        if ignores is None:
            ignores = np.ones_like(ns)
        for i, (ci, ni, igi) in enumerate(zip(cs, ns, ignores)):
            for j, (cj, nj, igj) in enumerate(zip(cs, ns, ignores)):
                if ci != cj:
                    var_num = min(ni, nj) * (4 * max(ni, nj) / N) * (1 - 4 * max(ni, nj) / N)
                    var_coef = ((ci - xmean) * (cj - xmean) - 2 * (ci - xmean) * (cs[r_max] - xmean) + (
                                cs[r_max] - xmean) ** 2) ** 2
                    var += var_coef * var_num * igi * igj
                else:
                    var_num = 2 * (ni - 1) * 4 * ni / N * (1 - (4 * (2 * ni - 1)) / (3 * N))
                    var_coef = (ci - cs[r_max]) ** 4
                    var += var_coef * var_num * igi * igj

                std_dict[(i, j)] = np.sqrt(var_num * igi * igj)
                std_coef_dict[(i, j)] = np.sqrt(var_coef)

        return np.sqrt(var), std_dict, std_coef_dict