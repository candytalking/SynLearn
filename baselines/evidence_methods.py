import warnings
import numpy as np
# from numba import njit


def each_evidence(y_, f, fh, v, s, vh, N, D):
    """
    compute the maximum evidence for each class
    """
    epsilon = 1e-5
    alpha = 1.0
    beta = 1.0
    lam = alpha / beta
    tmp = (vh @ (f @ np.ascontiguousarray(y_)))
    for _ in range(11):
        gamma = (s / (s + lam)).sum()
        m = v @ (tmp * beta / (alpha + beta * s))
        alpha_de = (m * m).sum()
        alpha = gamma / (alpha_de + epsilon)
        beta_de = ((y_ - fh @ m) ** 2).sum()
        beta = (N - gamma) / (beta_de + epsilon)
        new_lam = alpha / beta
        if np.abs(new_lam - lam) / lam < 0.01:
            break
        lam = new_lam
    evidence = D / 2.0 * np.log(alpha) \
               + N / 2.0 * np.log(beta) \
               - 0.5 * np.sum(np.log(alpha + beta * s)) \
               - beta / 2.0 * (beta_de + epsilon) \
               - alpha / 2.0 * (alpha_de + epsilon) \
               - N / 2.0 * np.log(2 * np.pi)
    return evidence / N, alpha, beta, m


class LogME(object):
    def __init__(self, regression=False):
        """
            :param regression: whether regression
        """
        self.regression = regression
        self.fitted = False
        self.reset()

    def reset(self):
        self.num_dim = 0
        self.alphas = []
        self.betas = []
        self.ms = []

    def fit(self, f: np.ndarray, y: np.ndarray):
        """
        :param f: [N, F], feature matrix from pre-trained model
        :param y: target labels.
            For classification, y has shape [N] with element in [0, C_t).
            For regression, y has shape [N, C] with C regression-labels
        :return: LogME score (how well f can fit y directly)
        """
        if self.fitted:
            warnings.warn('re-fitting for new data. old parameters cleared.')
            self.reset()
        else:
            self.fitted = True
        f = f.astype(np.float64)
        if self.regression:
            y = y.astype(np.float64)

        fh = f
        f = f.transpose()
        D, N = f.shape
        v, s, vh = np.linalg.svd(f @ fh, full_matrices=True)

        y_ = y
        evidence, alpha, beta, m = each_evidence(y_, f, fh, v, s, vh, N, D)
        return evidence

    def predict(self, f: np.ndarray):
        """
        :param f: [N, F], feature matrix
        :return: prediction, return shape [N, X]
        """
        if not self.fitted:
            raise RuntimeError("not fitted, please call fit first")
        f = f.astype(np.float64)
        logits = f @ self.ms.T
        if self.regression:
            return logits
        return np.argmax(logits, axis=-1)

    def probability(self, f: np.ndarray, y: np.ndarray):
        """
        :param f: [N, F], feature matrix
        :param y: target labels.
            For classification, y has shape [N] with element in [0, C_t).
            For regression, y has shape [N, C] with C regression-labels
        :return: probability score (how well each f can fit each y directly), return shape [N]
        """
        if not self.fitted:
            raise RuntimeError("not fitted, please call fit first")
        f = f.astype(np.float64)
        if self.regression:
            y = y.astype(np.float64)
        scores = []
        D = f.shape[1]
        for i in range(self.num_dim):
            alpha = self.alphas[i]
            beta = self.betas[i]
            y_ = y[:, i] if self.regression else (y == i).astype(np.float64)
            common_term = 0.5 * np.log(beta) + D / 2 * np.log(alpha) - 0.5 * np.log(2 * np.pi)
            beta_uTu = beta * np.sum((f * f), axis=1, keepdims=True)
            ms_de = alpha + beta_uTu
            ms = beta * y_.reshape(-1, 1) * f / ms_de
            logdetAs = (D - 1) * np.log(alpha) + np.log(ms_de.reshape(-1))
            err = (np.sum(ms * f, axis=1) - y_) ** 2
            ms_norm = np.sum(ms * ms, axis=1)
            ans = common_term - 0.5 * beta * err - 0.5 * alpha * ms_norm - 0.5 * logdetAs
            scores.append(ans)
        return np.mean(np.stack(scores), axis=0)


def LEEP_score(y_hat, y, p=None, p1=None):
    """
    :param y_hat: [N, C_f], model label distribution over |C_f|.
    :param y: target true labels.
        For classification only, y has shape [N] with element in [0, C_t).
    :return: a LEEP score, the larger the better.
    """
    if p is None or p1 is None:
        assert y_hat.shape[-2] == y.shape[-1]
        N = y_hat.shape[-2]
        C_f = y_hat.shape[-1]
        C_t = int(np.max(y))+1
        y_hat = y_hat.reshape([-1, N, C_f])
        M = y_hat.shape[0]
        sum_pc = np.sum(y_hat, axis=1, keepdims=True)
        pc = np.concatenate([np.sum(y_hat[:, y==i], axis=1, keepdims=True) for i in range(C_t)], axis=1)
        p = pc/(sum_pc + 1e-5)
        p1 = pc/(N/C_t)
        tmp_score = [np.log(np.sum(p[:, int(y[i])] * y_hat[:, i], axis=-1, keepdims=True)) for i in range(N)]
        total_score = np.sum(np.concatenate(tmp_score, axis=1), axis=1) / N
    else:
        p = np.stack(p)
        p1 = np.stack(p1)
        total_score = np.stack([np.mean(np.concatenate([np.log(np.sum(p[j, :, int(y[j][i])] * y_hat[j][:, i],
                                axis=-1, keepdims=True)) for i in range(y[j].shape[0])], axis=-1), axis=-1)
                                for j in range(len(y_hat))])
    return total_score, p, p1

def LEEP_score_list(y_hat, y):
    """
    :param y_hat: [N, C_f], model label distribution over |C_f|.
    :param y: target true labels.
        For classification only, y has shape [N] with element in [0, C_t).
    :return: a LEEP score, the larger the better.
    """
    assert len(y_hat) == len(y)
    total_score, p, p1 = [], [], []
    for task_idx in range(len(y_hat)):
        task_y_hat, task_y = np.concatenate(y_hat[task_idx], axis=1), y[task_idx]
        assert task_y_hat.shape[-2] == task_y.shape[-1]
        (M, N, C_f) = task_y_hat.shape
        C_t = int(np.max(task_y)) + 1
        sum_pc = np.sum(task_y_hat, axis=1, keepdims=True)
        pc = np.concatenate([np.sum(task_y_hat[:, task_y == i], axis=1, keepdims=True) for i in range(C_t)], axis=1)
        task_p = pc / (sum_pc + 1e-5)
        task_p1 = pc / np.reshape(np.asarray([sum(task_y==i) for i in range(C_t)]), [1, -1, 1])
        tmp_score = [np.log(np.sum(task_p[:, int(task_y[i])] * task_y_hat[:, i], axis=-1, keepdims=True)) for i in range(N)]
        task_score = np.sum(np.concatenate(tmp_score, axis=1), axis=1) / N
        total_score.append(task_score)
        p.append(task_p)
        p1.append(task_p1)
    return np.stack(total_score), np.stack(p), np.stack(p1)
    if y_hat.ndim == 2:
        total_score = sum([np.log(np.dot(p[int(y[i])], y_hat[i])) for i in range(N)])/N
    else:
        total_score = sum([np.log(p[int(y[i]), j]) for i in range(N) for j in range(C_f) if p[int(y[i]), j] > 0])/N
    if not classwise:
        return total_score
    else:
        return total_score, p, p1


def NCE_score(y_hat, y):
    """
    :param y_hat: [N], model label predictions with elements in [0, C_f).
    :param y: target true labels.
        For classification only, y has shape [N] with element in [0, C_t).
    :return: a NCE score, the larger the better
    """
    assert y_hat.shape[0] == y.shape[0]
    N, C_f, C_t = y_hat.shape[0], int(np.max(y_hat))+1, int(np.max(y))+1
    score, p1 = 0, np.zeros([C_f, C_t])
    for class_idx1 in range(C_f):
        sum_p = np.sum(np.where(y_hat == class_idx1, 1, 0))
        if sum_p > 0:
            for class_idx2 in range(C_t):
                p1[class_idx1, class_idx2] = np.sum(np.where(y_hat == class_idx1, 1, 0)
                                                    * np.where(y == class_idx2, 1, 0))/sum_p
                if p1[class_idx1, class_idx2] > 0:
                    score += np.log(p1[class_idx1, class_idx2]) * p1[class_idx1, class_idx2] * sum_p
    score/=N
    return score
