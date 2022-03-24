import numpy as np


class NNetwork:
    """Многослойная полносвязная нейронная сеть прямого распространения"""

    @staticmethod
    def getTotalWeights(*layers):
        return sum([(layers[i]+1)*layers[i+1] for i in range(len(layers)-1)])

    def __init__(self, inputs, *layers):
        self.layers = []        # список числа нейронов по слоям
        self.acts = []          # список функций активаций (по слоям)

        # формирование списка матриц весов для нейронов каждого слоя и списка функций активации
        self.n_layers = len(layers)
        for i in range(self.n_layers):
            self.acts.append(self.act_relu)
            if i == 0:
                self.layers.append(self.getInitialWeights(layers[0], inputs+1))         # +1 - это вход для bias
            else:
                self.layers.append(self.getInitialWeights(layers[i], layers[i-1]+1))    # +1 - это вход для bias

        self.acts[-1] = self.act_th     #последний слой имеет пороговую функцию активакции

    def getInitialWeights(self, n, m):
        return np.random.triangular(-1, 0, 1, size=(n, m))

    def act_relu(self, x):
        x[x < 0] = 0
        return x

    def act_th(self, x):
        x[x > 0] = 1
        x[x <= 0] = 0
        return x

    def get_weights(self):
        return np.hstack([w.ravel() for w in self.layers])

    def set_weights(self, weights):
        off = 0
        for i, w in enumerate(self.layers):
            w_set = weights[off:off+w.size]
            off += w.size
            self.layers[i] = np.array(w_set).reshape(w.shape)

    def predict(self, inputs):
        f = inputs
        for i, w in enumerate(self.layers):
            f = np.append(f, 1.0)       # добавляем значение входа для bias
            f = self.acts[i](w @ f)

        return f


# net = NNetwork(4, 4, 1)
# print(net.predict([1, 0, 1, 1]))
# net.set_weights(net.get_weights())
# print(int(net.predict([1, 0, 1, 1])))
