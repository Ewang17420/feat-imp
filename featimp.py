import numpy as np
import copy
from scipy.stats import spearmanr


class SpearFeatureImportance():
    def __init__(self):
        pass

    def fit(self, X, y):
        self.X = X.transpose()
        self.y = y

        # build a Spearman's array for searching
        self.spear_list = []
        for i in range(len(self.X)):
            score, _ = spearmanr(self.X[i], self.y.reshape(-1, 1))
            self.spear_list.append(score)

    def mRMR(self, k):
        """

        :param X: features to select, type nxm np array
        :param y: labels , type 1d np array
        :param k: number of features to select, int
        :return: features after selected.
        """
        # # build a Spearman's array for searching
        # spear_list = []
        # for i in range(len(X)):
        #     score, _ = spearmanr(X[i], self.y.reshape(-1, 1))
        #     spear_list.append(score)

        n, m = self.X.shape
        selected = []
        not_selected = np.arange(n)

        s2_list = np.zeros(n)
        for _ in range(k):
            scores = np.zeros(n)
            for col in not_selected:
                score1 = self.spear_list[col]
                if selected:
                    for c in selected:
                        s, _ = spearmanr(self.X[selected[-1]], self.X[col])
                        s2_list[c] += s
                    score2 = s2_list[c] / len(selected)
                    scores[col] = score1 - score2
                else:
                    scores[col] = score1
            best_col = list(scores).index(np.max(scores[np.nonzero(scores)]))
            selected.append(best_col)
            not_selected = np.delete(not_selected, np.where(not_selected == best_col))

        result = []
        for select in selected:
            result.append(self.X[select])
        result = np.array(result)
        return selected, result

    def score(self):

        return self.spear_list

def dropcol_importances(model, metric, X, y):
    model.fit(X, y)
    baseline = metric(y, model.predict(X))
    result = []
    for col in X.columns:
        X_drop = X.drop(col, axis=1)
        model_new = copy.deepcopy(model)
        model_new.fit(X_drop, y)
        m = metric(y, model_new.predict(X_drop))
        result.append(baseline - m)
    return result

def permutation_importances(model, metric, X, y):
    model.fit(X, y)
    baseline = metric(y, model.predict(X))
    result = []
    for col in X.columns:
        save = X[col].copy()
        X[col] = np.random.permutation(X[col])
        m = metric(y, model.predict(X))
        X[col] = save
        result.append(baseline - m)
    return result