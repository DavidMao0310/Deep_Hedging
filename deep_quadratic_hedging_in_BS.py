import numpy as np
from tensorflow.keras.layers import Input, Dense, Concatenate, Multiply, Lambda, Add, Dot, Subtract
from tensorflow.keras.models import Model
from tensorflow.keras import initializers
from tensorflow.keras import backend
import matplotlib.pyplot as plt
import scipy


# Need to change V0, mu ,r ...

def BS(S0, strike, T, sigma):
    return S0 * scipy.stats.norm.cdf(
        (np.log(S0 / strike) + 0.5 * T * sigma ** 2) / (np.sqrt(T) * sigma)) - \
           strike * scipy.stats.norm.cdf(
        (np.log(S0 / strike) - 0.5 * T * sigma ** 2) / (np.sqrt(T) * sigma))


class Deep_Hedging_NN_Set:
    def __init__(self, dim_of_price, num_of_layers, num_of_nodes, num_time_discretion, S0, T, r, strike, sigma, mu,
                 xtrain, ytrain, grid):
        self.m = dim_of_price
        self.d = num_of_layers
        self.n = num_of_nodes
        self.N = num_time_discretion
        self.S0 = S0
        self.T = T
        self.r = r
        self.sigma = sigma
        self.mu = mu
        self.strike = strike
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.grid = grid
        self.layers = []

    def custom_loss(self, y_true, y_pred):
        # return losses.mean_squared_error(y_true[0], y_pred[0])
        z = y_pred[:, 0] - y_true[:, 0]
        z = backend.mean(backend.square(z))
        return z

    def build_layers(self):
        for j in range(self.N):
            for i in range(self.d):
                if i < self.d - 1:
                    nodes = self.n
                    layer = Dense(nodes, activation='tanh', trainable=True,
                                  kernel_initializer=initializers.RandomNormal(0, 1),
                                  # kernel_initializer='random_normal',
                                  bias_initializer='random_normal',
                                  name=str(i) + str(j))
                else:
                    nodes = self.m
                    layer = Dense(nodes, activation='linear', trainable=True,
                                  kernel_initializer=initializers.RandomNormal(0, 0.1),
                                  # kernel_initializer='random_normal',
                                  bias_initializer='random_normal',
                                  name=str(i) + str(j))
                self.layers = self.layers + [layer]

    def train_model(self):
        # Implementing the loss function
        # Inputs is the training set below, containing the price S0,
        # the initial hedging being 0, and the increments of the log price process
        price = Input(shape=(self.m,))
        hedge = Input(shape=(self.m,))
        hedgeeval = Input(shape=(self.m,))
        V0 = Input(shape=(self.m,))
        inputs = [price] + [hedge] + [hedgeeval] + [V0]
        outputhelper = []

        for j in range(self.N):
            strategy = price
            strategyeval = hedgeeval
            for k in range(self.d):
                # strategy at j is the hedging strategy at j , i.e. the neural network g_j.
                # It is applied to the price here so that we get the actual value.
                strategy = self.layers[k + (j) * self.d](strategy)
                # we also store the same neural network in functional form in order to access it later.
                strategyeval = self.layers[k + (j) * self.d](strategyeval)
            incr = Input(shape=(self.m,))
            logprice = Lambda(lambda x: backend.log(x))(price)
            logprice = Add()([logprice, incr])
            # creating the price at time j+1 from log price at time j and log price increment at time j+1
            pricenew = Lambda(lambda x: backend.exp(x))(logprice)
            priceincr = Subtract()([pricenew, price])
            hedgenew = Multiply()([strategy, priceincr])
            hedge = Add()([hedge, hedgenew])  # building up the discretized stochastic integral
            inputs = inputs + [incr]
            outputhelper = outputhelper + [strategyeval]
            price = pricenew
        payoff = Lambda(lambda x: 0.5 * (backend.abs(x - self.strike) + x - self.strike))(price)
        outputs = Subtract()([payoff, hedge])
        outputs = Subtract()([outputs, V0])  # payoff minus price minus hedge
        outputs = [outputs] + outputhelper + [V0]  # hedge PnL [0], neural networks [1:100], initial capital V0 [101].
        outputs = Concatenate()(outputs)  # Concatenate layer turns lists into tensor
        # We use the Functional API in Tensorflow to implement the neural network that maps inputs to outputs.
        self.model_hedge_strategy = Model(inputs=inputs, outputs=outputs)
        self.model_hedge_strategy.compile(optimizer='adam', loss=self.custom_loss)
        for i in range(5):
            self.model_hedge_strategy.fit(x=self.xtrain, y=self.ytrain, epochs=1, verbose=True, batch_size=1000)

    def model_eval(self, xtest, num_of_test):
        l = 60
        s = np.linspace(0.5, 1.5, num_of_test)
        # Black Scholes delta hedge of call
        option_delta = scipy.stats.norm.cdf(
            (np.log(s / self.strike) + 0.5 * (self.T - self.grid[l]) * self.sigma ** 2) / (
                    np.sqrt(self.T - self.grid[l]) * self.sigma))
        # neural network g_l
        y = self.model_hedge_strategy.predict(xtest)[:, l]
        plt.plot(s, y, label='predict')
        plt.plot(s, option_delta, label='option_delta')
        plt.legend()
        plt.tight_layout()
        # plt.plot(s,y,s,y_mu,s,z) #Uncomment this for the second neural network, where mu is not equal 0.
        plt.show()


N = 100  # time discretion
S0 = 1  # initial value of the asset
T = 1  # maturity
r = 0  # interest rate
strike = 1.0  # f(S)=(S-1)_+ European Call Contract
sigma = 0.2  # volatility in Black Scholes
mu = 0.1
m = 1  # dimension of price
d = 3  # number of layers in strategy
n = 32  # nodes in the first but last layers
priceBS = BS(S0, strike, T, sigma)
print('Price of a Call option in the Black scholes model with initial price', S0, 'strike', strike, 'maturity', T,
      'and volatility', sigma, 'is equal to', np.round(BS(S0, strike, T, sigma), 2))

grid = [(i / N) * T for i in range(N + 1)]  # crete the time discretization
Ktrain = 10 ** 6
Ktest = 50
initialprice = S0
# xtrain consists of the price S0, the initial hedging being 0, and the increments of the log price process
xtrain = ([initialprice * np.ones((Ktrain, m))] +
          [np.zeros((Ktrain, m))] +
          [np.ones((Ktrain, m))] +
          [priceBS * np.ones((Ktrain, m))] +
          [np.random.normal(-(sigma) ** 2 / 2 * (grid[i + 1] - grid[i]), sigma * np.sqrt(grid[i + 1] - grid[i]),
                            (Ktrain, m)) for i in range(N)])
# y is the hedge error, should be 0
ytrain = np.zeros((Ktrain, 1 + N))
xtest = ([initialprice * np.ones((Ktest, m))] +
         [np.zeros((Ktest, m))] +
         [0.5 * np.ones((Ktest, m)) + np.cumsum(np.ones((Ktest, m)) * (1.5 - 0.5) / Ktest,
                                                axis=0)] +
         # This creates the grid to evaluate the neural networks via the input hedgeeval for
         # the comparison with the BS delta hedge. Change this if you go to higher dimensions
         [priceBS * np.ones((Ktest, m))] +
         [np.random.normal(-(sigma) ** 2 / 2 * (grid[i + 1] - grid[i]), sigma * np.sqrt(grid[i + 1] - grid[i]),
                           (Ktest, m)) for i in range(N)])

hedge_NN = Deep_Hedging_NN_Set(m, d, n, N, S0, T, r, strike, sigma, mu, xtrain, ytrain, grid)
hedge_NN.build_layers()
hedge_NN.train_model()
hedge_NN.model_eval(xtest, Ktest)
