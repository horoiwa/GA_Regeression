import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
import random

from deap import algorithms
from deap import base
from deap import creator
from deap import tools


class RidgeGA(object):
    """ GAによる特徴量選択
        まずは単純にonemax問題を
    """
    def __init__(self, X, y, n_gen):
        self.X = X
        self.y = y

        self.n_eval = 10
        self.weights = (-1.0, 1.0)
        self.n_gen = n_gen

        self.pop = None
        self.log = None
        self.hof = None
        self.result = None

        self._MAX_FEATURES = len(self.X.columns)
        self._DESIRED_FEATURES = self._MAX_FEATURES//2

    def eval_score(self, X, n):
        """ RidgeCV
            Parameters
            -------------
            X: pandas dataframe
            n: train_test_splitの回数

            Return
            -------------
            score: average score
        """
        scores = []
        for _ in range(n):
            X_train, X_test, y_train, y_test = train_test_split(X, self.y, test_size=0.4) 
            model = RidgeCV()
            model.fit(X_train, y_train)
            scores.append(model.score(X_test, y_test))

        score = np.array(scores).mean()
        return score

    def run(self):
        """ Feature optimization by NSGA-2
            max_item means max_feature
        """
        def evalIndividual(individual):
            n_features = sum(individual)
            
            if n_features == 0:
                return 9999, -9999
            elif n_features > DESIRED_FEATURES:
                return 9999, -9999
            else:
                X_temp = self.X.iloc[:, [bool(val) for val in individual]]
                score = self.eval_score(X_temp, self.n_eval)

            # print(n_features, " ", score)
            return n_features, score

        def main():
            NGEN = self.n_gen
            MU = 100
            LAMBDA = 400
            CXPB = 0.7
            MUTPB = 0.1

            pop = toolbox.population(n=MU)
            hof = tools.ParetoFront()
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean, axis=0)
            stats.register("std", np.std, axis=0)
            stats.register("min", np.min, axis=0)
            stats.register("max", np.max, axis=0)
           
            pop, log = algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB,
                                                 MUTPB, NGEN, stats, halloffame=hof)

            return pop, log, hof

        MAX_FEATURES = self._MAX_FEATURES
        DESIRED_FEATURES = self._DESIRED_FEATURES

        #: 特徴数を最小化　精度を最大化
        creator.create("Fitness", base.Fitness, weights=self.weights)   
        creator.create("Individual", list, fitness=creator.Fitness)

        toolbox = base.Toolbox()
        toolbox.register("attr_bool", random.randint, 0, 1)
        toolbox.register("individual", tools.initRepeat, creator.Individual, 
                         toolbox.attr_bool, MAX_FEATURES)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("evaluate", evalIndividual)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
        toolbox.register("select", tools.selNSGA2)

        self.pop, self.log, self.hof = main()
        self.result = self.create_result()

    def create_result(self):
        scores = []
        n_features = []
        for ind in self.hof:
            n_features.append(sum(ind))
            X_temp = self.X.iloc[:, [bool(val) for val in ind]]
            score = self.eval_score(X_temp, 200)
            scores.append(score)

        X = pd.DataFrame(np.array(self.hof), columns=self.X.columns) 
        scores = pd.DataFrame(np.array(scores), columns=["SCORE"])
        n_features = pd.DataFrame(np.array(n_features), columns=["N_feature"])
        result = pd.concat([scores, n_features, X], 1)
        return result
