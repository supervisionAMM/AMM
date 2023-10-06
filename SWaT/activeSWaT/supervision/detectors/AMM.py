import random
import numpy as np

from sklearn.linear_model import LinearRegression
from scipy import stats
from cvxopt import matrix, solvers

np.set_printoptions(suppress=True, precision=10, threshold=2000, linewidth=150)
solvers.options['maxiters'] = 10
solvers.options['show_progress'] = False


class AMM:
    # Active Monitoring Mechanism

    def __init__(self, logging, auxiliary_signal_trigger_mode, auxiliary_signal_design_mode):

        self.logging = logging
        self.counting = 0

        # "combine_trigger"/"prior_trigger"/"posterior_trigger"/"timing_trigger"
        self.auxiliary_signal_trigger_mode = auxiliary_signal_trigger_mode

        # "optimal_design"/"random_design"
        self.auxiliary_signal_design_mode = auxiliary_signal_design_mode

        ############################## knowledge section ##################################################
        # nominal model parameters
        ## x(k) = A*x(k-1) + B*u(k-1)
        ## y(k) = C*x(k)
        self.A = np.zeros((3, 3))

        self.B = np.array([[0.4259, -0.3703, 0.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.3703, -0.3704, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0, 0.3702, -0.0370]])

        self.C = 1.0 * np.eye(3)

        # deviated model parameter
        self.B_k_prior = np.copy(self.B.T)  # using copy to avoid B_k_prior and B_k_posterior change together
        self.P_k_prior = 5.0e-06 * np.eye(6)  # estimate uncertainty
        self.B_k_posterior = np.copy(self.B.T)
        self.P_k_posterior = 5.0e-06 * np.eye(6)
        self.B_k_predict = np.copy(self.B.T)
        self.P_k_predict = 5.0e-06 * np.eye(6)

        # system state
        self.state_prior = np.zeros((3, 3))
        self.state_var_prior = np.zeros((3, 3))
        self.state_posterior = np.zeros((3, 3))
        self.state_var_posterior = np.zeros((3, 3))

        # uncertainty of model parameter value
        self.Q = 5.0e-06 * np.eye(6)  # process noise

        # uncertainty compensation terms
        ## x(k) = A*x(k-1) + B*u(k-1) + w(k)
        ## y(k) = C*x(k) + v(k)
        ## -->
        ## y(k) = C*A*x(k-1) + C*B*u(k-1) + C*w(k) + v(k)
        self.gamma = np.zeros((3, 3))

        self.W = np.zeros((3, 3))  # measurement error of environment input

        self.V = np.array([[0.0008, 0.0, 0.0],
                           [0.0, 0.0008, 0.0],
                           [0.0, 0.0, 0.0008]])

        ############################## passive decision section ##################################################
        # safe region of model parameter values
        self.percent = [[0.35, 0.35], [0.35, 0.35],
                        [0.35, 0.35], [0.35, 0.35],
                        [0.35, 0.35], [3.50, 3.50]]

        self.safe_region = np.array([[self.B[0][0] - self.percent[0][0] * abs(self.B[0][0]),
                                      self.B[0][0] + self.percent[0][1] * abs(self.B[0][0])],
                                     [self.B[0][1] - self.percent[1][0] * abs(self.B[0][1]),
                                      self.B[0][1] + self.percent[1][1] * abs(self.B[0][1])],

                                     [self.B[1][2] - self.percent[2][0] * abs(self.B[1][2]),
                                      self.B[1][2] + self.percent[2][1] * abs(self.B[1][2])],
                                     [self.B[1][3] - self.percent[3][0] * abs(self.B[1][3]),
                                      self.B[1][3] + self.percent[3][1] * abs(self.B[1][3])],

                                     [self.B[2][4] - self.percent[4][0] * abs(self.B[2][4]),
                                      self.B[2][4] + self.percent[4][1] * abs(self.B[2][4])],
                                     [self.B[2][5] - self.percent[5][0] * abs(self.B[2][5]),
                                      self.B[2][5] + self.percent[5][1] * abs(self.B[2][5])]])
        # print("safe_region:\n{}\n".format(self.safe_region))

        # probability threshold
        self.prob_threshold = 0.999999998  # 6sigma

        # alarm signal
        self.alarm = np.zeros(6)

        # historical measurements
        self.list_u = []  # historical controller output
        self.list_y = []  # historical managed system output

        ############################## auxiliary signal trigger section ################################################
        # index range
        self.index_range = [0, 6]

        # active flag
        self.active_flag = np.zeros(6)

        # 1) auxiliary_signal_trigger_mode = "combine_trigger"/"prior_trigger"/"posterior_trigger"
        self.prediction_sw = 2

        ## use for model prediction
        self.list_B_k_posterior = [[self.B_k_posterior[0][0]], [self.B_k_posterior[1][0]],
                                   [self.B_k_posterior[2][1]], [self.B_k_posterior[3][1]],
                                   [self.B_k_posterior[4][2]], [self.B_k_posterior[5][2]]]

        self.list_P_k_posterior = [[self.P_k_posterior[0][0]], [self.P_k_posterior[1][1]],
                                   [self.P_k_posterior[2][2]], [self.P_k_posterior[3][3]],
                                   [self.P_k_posterior[4][4]], [self.P_k_posterior[5][5]]]

        self.list_B_k_predict = [[self.B_k_posterior[0][0]], [self.B_k_posterior[1][0]],
                                 [self.B_k_posterior[2][1]], [self.B_k_posterior[3][1]],
                                 [self.B_k_posterior[4][2]], [self.B_k_posterior[5][2]]]

        self.list_P_k_predict = [[self.P_k_posterior[0][0]], [self.P_k_posterior[1][1]],
                                 [self.P_k_posterior[2][2]], [self.P_k_posterior[3][3]],
                                 [self.P_k_posterior[4][4]], [self.P_k_posterior[5][5]]]

        self.prediction_B_model = [[1.0000], [1.0000], [1.0000], [1.0000], [1.0002], [0.9982]]
        self.prediction_P_model = [[1.0003], [1.0004], [1.0003], [1.0003], [1.0001], [1.0003]]

        self.active_trigger_threshold = 0.999999998  # 6sigma

        # 2) auxiliary_signal_trigger_mode = "timing_trigger"
        self.supervision_time_interval = 1.0
        self.timing_interval = 60
        self.timer = np.zeros(6)

        ############################## auxiliary signal design section ##################################################
        # 1) auxiliary_signal_design_mode = "optimal_design"/"random_design"
        ## constraints
        self.safety_n_sigma = 6
        self.U_constraint = [[0.0, 1.0], [0.0, 1.0],
                             [0.0, 1.0], [0.0, 1.0],
                             [0.0, 1.0], [0.0, 1.0]]
        self.Y_constraint = [[250, 1100], [250, 1200], [250, 1200]]

        ## safety auxiliary signal constraints
        self.safety_AS = np.copy(self.U_constraint)

        ## cost function parameters of the auxiliary signal
        self.T = 1.0
        self.r_C = [1.5, 1.5,
                    1.5, 1.5,
                    1.5, 1.5]

        ## safety function parameters of auxiliary signal: 2.0/(LIT_low + LIT_high)
        self.r_S = [2.0 / (250 + 1100), 2.0 / (250 + 1100),
                    2.0 / (250 + 1200), 2.0 / (250 + 1200),
                    2.0 / (250 + 1200), 2.0 / (250 + 1200)]

        ## accuracy enhancement function parameters of auxiliary signal
        self.r_D = [1.0 / abs(self.B[0][0]), 1.0 / abs(self.B[0][1]),
                    1.0 / abs(self.B[1][2]), 1.0 / abs(self.B[1][3]),
                    1.0 / abs(self.B[2][4]), 1.0 / abs(self.B[2][5])]

        # optimal auxiliary control signal
        self.opt_AS = np.zeros(6)

        # 2) auxiliary_signal_design_mode = "random_design"
        self.rnd_seed = 1

    def deviation_detector(self, u_1, y, LIT):

        self.counting = self.counting + 1

        if len(self.list_u) < 2:
            self.list_u.append(u_1)
            self.list_y.append(y)
        else:
            self.list_u.append(u_1)
            self.list_y.append(y)

            if self.logging:
                print("\n############### deviation_detector k=" + str(self.counting) + " ############### ")

            # state estimation
            ## input: time series observation
            ##  [0]     [1]     [2]
            ## u(k-3), u(k-2), u(k-1)
            ## y(k-2), y(k-1), y(k)
            y_2 = self.list_y[0]
            u_2 = self.list_u[1]

            ## output: state_prior, state_var_prior
            self.state_posterior, \
                self.state_var_posterior, \
                self.state_prior, \
                self.state_var_prior = self.observer(self.B_k_posterior,
                                                     self.state_prior,
                                                     self.state_var_prior,
                                                     y_2, u_2)

            # model parameter estimation and deviation detection
            ## input: time series observation
            u_1 = self.list_u[2]
            y_0 = self.list_y[2]

            ## output: B_k_posterior, P_k_posterior
            self.B_k_prior, \
                self.P_k_prior, \
                self.B_k_posterior, \
                self.P_k_posterior = self.estimator(self.state_prior,
                                                    self.state_var_prior,
                                                    self.B_k_prior,
                                                    self.P_k_prior,
                                                    self.B_k_posterior,
                                                    self.P_k_posterior,
                                                    u_1, y_0)

            for index in range(self.index_range[0], self.index_range[1]):
                if abs(u_1[index][int(index / 2)]) > 0.0:
                    # passive decision
                    self.alarm[index] = self.passive_alarmer(index,
                                                             self.B_k_posterior[index][int(index / 2)],
                                                             self.P_k_posterior[index][index])

                    # reset auxiliary signal trigger and design
                    self.active_flag[index] = 0
                    self.opt_AS[index] = 0

                    # if self.logging:
                    #     print("********** passive decision index={" + str(index) + "} **********")
                    #     print("u_1[{}]:{}".format(index, u_1[index][int(index / 2)]))
                    #     print("y_0[{}]:{}".format(index, y_0[int(index / 2)][int(index / 2)]))
                    #     print("B_k_prior[{}]:{}".format(index, self.B_k_prior[index][int(index / 2)]))
                    #     print("P_k_prior[{}]:{}".format(index, self.P_k_prior[index][index]))
                    #     print("B_k_posterior[{}]:{}".format(index, self.B_k_posterior[index][int(index / 2)]))
                    #     print("P_k_posterior[{}]:{}".format(index, self.P_k_posterior[index][index]))
                    #     print("alarm[{}]:{}\n".format(index, self.alarm[index]))

                    ## store historical model parameters
                    self.list_B_k_posterior[index].append(self.B_k_posterior[index][int(index / 2)])
                    self.list_P_k_posterior[index].append(self.P_k_posterior[index][index])
                    self.list_B_k_predict[index].append(self.B_k_posterior[index][int(index / 2)])
                    self.list_P_k_predict[index].append(self.P_k_posterior[index][index])

                    # remove first point
                    if len(self.list_B_k_posterior[index]) > 2.0 * (self.prediction_sw - 1):
                        del (self.list_B_k_posterior[index][0])
                        del (self.list_P_k_posterior[index][0])

                    if len(self.list_B_k_predict[index]) > 2.0 * (self.prediction_sw - 1):
                        del (self.list_B_k_predict[index][0])
                        del (self.list_P_k_predict[index][0])

                else:
                    # when no alarm, continue active part
                    if np.all(self.alarm == 0):
                        # auxiliary signal trigger
                        if self.auxiliary_signal_trigger_mode != "timing_trigger":

                            if self.logging:
                                print("********** optimal_trigger index={" + str(index) + "} **********")

                            if len(self.list_B_k_posterior[index]) >= 2.0 * (self.prediction_sw - 1):
                                self.update_prediction_model(index)

                            self.active_flag[index] = \
                                self.activate_optimal_trigger(index,
                                                              self.auxiliary_signal_trigger_mode,
                                                              self.B_k_prior[index][int(index / 2)],
                                                              self.P_k_prior[index][index],
                                                              self.prediction_B_model[index],
                                                              self.prediction_P_model[index])

                        elif self.auxiliary_signal_trigger_mode == "timing_trigger":
                            self.active_flag[index] = self.activate_timing_trigger(index)

                            if self.logging:
                                print("********** timing_trigger index={" + str(index) + "} **********")
                                print("active_flag[{}]:{}\n".format(index, self.active_flag[index]))

                        # auxiliary signal design
                        if self.active_flag[index] == 1:
                            ## safety auxiliary signal
                            self.safety_AS[index] = self.calculate_safety_AS(index,
                                                                             u_1[index][int(index / 2)],
                                                                             LIT[int(index / 2)],
                                                                             self.B_k_prior[index][int(index / 2)],
                                                                             self.P_k_prior[index][index])

                            ## optimal auxiliary signal
                            if self.auxiliary_signal_design_mode == "optimal_design":
                                if self.logging:
                                    print("********** optimal_design index={" + str(index) + "} **********")

                                self.opt_AS[index] = \
                                    self.calculate_optimal_AS(index, self.B_k_posterior[index][int(index / 2)])

                            elif self.auxiliary_signal_design_mode == "random_design":
                                self.opt_AS[index] = self.calculate_random_AS(index)

                                if self.logging:
                                    print("********** random_design index={" + str(index) + "} **********")
                                    print("safety_AS[{}]:{}".format(index, self.safety_AS[index]))
                                    print("opt_AS[{}]:{}\n".format(index, self.opt_AS[index]))

            # # degradation detector
            # ## tank101
            if u_1[0][0] == 0 and u_1[1][0] == 0 and self.alarm[0] == self.alarm[1] == 0:
                delta = y_0[0][0]
                self.alarm[0] = self.alarm[1] = self.degradation_detector(self.V[0][0], delta)

            ## tank301
            if u_1[2][1] == 0 and u_1[3][1] == 0 and self.alarm[2] == self.alarm[3] == 0:
                delta = y_0[1][1]
                self.alarm[2] = self.alarm[3] = self.degradation_detector(self.V[1][1], delta)

            ## tank401
            if u_1[4][2] == 0 and u_1[5][2] == 0 and self.alarm[4] == self.alarm[5] == 0:
                delta = y_0[2][2]
                self.alarm[4] = self.alarm[5] = self.degradation_detector(self.V[2][2], delta)

            # update historical measurements
            self.list_u.remove(self.list_u[0])
            self.list_y.remove(self.list_y[0])

        return self.alarm, self.active_flag, self.opt_AS

    def observer(self, B_k_posterior, state_prior, state_var_prior, y_2, u_2):
        # refined nominal model
        ## y(k-2) = C*x(k-2) + v(k-2)
        ## x(k-1) = A*x(k-2) + B*u(k-2) + w(k-1)

        # update process
        ## observation model which maps the true state space into the observed space
        H = self.C

        ## measurement pre-fit residual
        R_k = y_2 - np.dot(H, state_prior)

        ## pre-fit residual variance
        S_k = np.dot(np.dot(H, state_var_prior), H) + self.V

        ## updated Kalman gain
        K = np.dot(np.dot(state_var_prior, H.T), np.linalg.inv(S_k))

        ## updated (a posteriori) state estimate
        state_posterior = state_prior + np.dot(K, R_k)

        ## updated (a posteriori) state estimate variance
        state_var_posterior = np.dot((np.eye(len(state_var_prior)) - np.dot(K, H)), state_var_prior)

        # prediction process
        ## predicted (a priori) state estimate
        state_prior = np.dot(self.A, state_posterior) + np.dot(B_k_posterior.T, u_2)

        ## predicted (a priori) state estimate variance
        state_var_prior = np.dot(np.dot(self.A, state_var_posterior), self.A) + self.W

        return state_posterior, state_var_posterior, state_prior, state_var_prior

    def estimator(self, state_prior, state_var_prior, B_k_prior, P_k_prior, B_k_posterior, P_k_posterior, u_1, y_0):
        # refined nominal model
        ## B(k) = B(k-1) + q(k)
        ## y(k) = C*A*x(k) + C*B(k)*u(k-1) + C*w(k) + v(k)

        # prediction process
        for index in range(0, 6):
            if abs(u_1[index][int(index / 2)]) > 0.0:
                ## predicted (a priori) model parameter estimate
                B_k_prior[index][int(index / 2)] = B_k_posterior[index][int(index / 2)]

                ## predicted (a priori) model parameter estimate variance
                P_k_prior[index][index] = P_k_posterior[index][index] + self.Q[index][index]

            else:
                ## keep last B_prior
                B_k_prior[index][int(index / 2)] = B_k_prior[index][int(index / 2)]

                ## enlarge P_k_prior as time goes on
                P_k_prior[index][index] = P_k_prior[index][index] + self.Q[index][index]

        # update process
        ## observation model which maps the true model parameter space into the observed space
        H = np.dot(self.C, u_1.T)
        # print("H:\n{}".format(H))

        ## measurement pre-fit residual
        error = y_0 - (np.dot(np.dot(self.C, self.A), state_prior) + np.dot(H, B_k_prior))
        # print("error:\n{}".format(error))

        ## pre-fit residual variance
        S_k = np.dot(np.dot(np.dot(self.C, self.A), state_var_prior), np.dot(self.C, self.A).T) + \
              np.dot(np.dot(H, P_k_prior), H.T) + np.dot(np.dot(self.C, self.W), self.C.T) + self.V
        # print("S_k:\n{}".format(S_k))

        ## updated Kalman gain
        K = np.dot(np.dot(P_k_prior, H.T), np.linalg.inv(S_k))
        # print("K:\n{}".format(K))

        ## updated (a posteriori) model parameter estimate and its variance or keep the last values
        for index in range(0, 6):

            if abs(u_1[index][int(index / 2)]) > 0.0:
                ## updated (a posteriori) model parameter estimate
                B_k_posterior[index][int(index / 2)] = B_k_prior[index][int(index / 2)] + \
                                                       K[index][int(index / 2)] * error[int(index / 2)][int(index / 2)]

                ## updated (a posteriori) model parameter estimate variance
                P_k_posterior[index][index] = (1.0 - np.dot(K, H)[index][index]) * P_k_prior[index][index]

            else:
                ## keep the last model parameter estimate and its variance
                B_k_posterior[index][int(index / 2)] = B_k_posterior[index][int(index / 2)]
                P_k_posterior[index][index] = P_k_posterior[index][index]

        return B_k_prior, P_k_prior, B_k_posterior, P_k_posterior

    def passive_alarmer(self, index, B_k_posterior, P_k_posterior):
        # alarm if the derived probability that model parameter value
        # falls into safe region exceeds the probability threshold

        # mean and variance
        loc = B_k_posterior
        scale = np.sqrt(P_k_posterior)

        cdf = stats.norm.cdf(self.safe_region[index][1], loc, scale) - \
              stats.norm.cdf(self.safe_region[index][0], loc, scale)

        passive_alarm = 0
        if cdf < self.prob_threshold:
            passive_alarm = 1

        return passive_alarm

    def degradation_detector(self, V, delta):
        # alarm if the derived probability that model parameter value
        # falls into safe region does not exceed the probability threshold

        loc = 0.0
        scale = np.sqrt(V)

        if delta <= loc:
            cdf = stats.norm.cdf(delta, loc, scale)
        else:
            cdf = 1.0 - stats.norm.cdf(delta, loc, scale)

        alarm = 0
        if cdf < (1.0 - self.prob_threshold) / 2:
            alarm = 1

        return alarm

    def update_prediction_model(self, index):

        list_B_k_posterior_i = self.list_B_k_posterior[index]
        list_P_k_posterior_i = self.list_P_k_posterior[index]
        sliding_window = self.prediction_sw

        X1 = []
        X2 = []
        Y1 = []
        Y2 = []
        data_slot_num = int(len(list_B_k_posterior_i) / sliding_window)
        for n in range(0, data_slot_num):
            X1.append([])
            X2.append([])

            for i in range(0, self.prediction_sw - 1):
                B_k = list_B_k_posterior_i[sliding_window * n + i]
                P_k = list_P_k_posterior_i[sliding_window * n + i]
                X1[n].append(B_k)
                X2[n].append(P_k)

            B_k = list_B_k_posterior_i[sliding_window * n + sliding_window - 1]
            P_k = list_P_k_posterior_i[sliding_window * n + sliding_window - 1]
            Y1.append(B_k)
            Y2.append(P_k)

        # print("X1: {}".format(X1))
        # print("Y1: {}".format(Y1))
        # print("X2: {}".format(X2))
        # print("Y2: {}".format(Y2))

        model_B_k = LinearRegression(fit_intercept=False)
        model_B_k.fit(X1, Y1)

        list_a_i = []
        for i in range(0, sliding_window - 1):
            a_i = round(model_B_k.coef_[0], 5)
            list_a_i.append(a_i)
            # print("a_{}: {}".format(i + 1, a_i))

        model_P_k = LinearRegression(fit_intercept=False)
        model_P_k.fit(X2, Y2)

        list_b_i = []
        for i in range(0, sliding_window - 1):
            b_i = round(model_P_k.coef_[0], 5)
            list_b_i.append(b_i)
            # print("b_{}: {}".format(i + 1, b_i))

        self.prediction_B_model[index] = list_a_i
        self.prediction_P_model[index] = list_b_i

        # if self.logging:
        #     print("update_prediction_model...")
        #     print("prediction_B_model[{}]: {}".format(index, self.prediction_B_model[index]))
        #     print("prediction_B_model[{}]: {}\n".format(index, self.prediction_B_model[index]))

    def activate_optimal_trigger(self, index, trigger_mode,
                                 B_k_prior_i, P_k_prior_i, prediction_B_model, prediction_P_model):

        prior_active_flag = 0
        if trigger_mode == "prior_trigger" or trigger_mode == "combine_trigger":
            # calculate the probability of model parameter value falls into safe region
            loc = B_k_prior_i
            scale = np.sqrt(P_k_prior_i)
            prior_trigger_indicator = stats.norm.cdf(self.safe_region[index][1], loc, scale) - \
                                      stats.norm.cdf(self.safe_region[index][0], loc, scale)

            if prior_trigger_indicator < self.active_trigger_threshold:
                prior_active_flag = 1

        predict_active_flag = 0
        if trigger_mode == "posterior_trigger" or trigger_mode == "combine_trigger":
            if len(self.list_B_k_predict[index]) > 0 and len(self.list_P_k_predict[index]) > 0:
                B_k_predict_i = 0.0
                end_i = len(self.list_B_k_predict[index]) - 1
                for i in range(0, len(prediction_B_model)):
                    B_k_predict_i = B_k_predict_i + prediction_B_model[i] * self.list_B_k_predict[index][end_i - i]

                P_k_predict_i = 0.0
                for i in range(0, len(prediction_P_model)):
                    P_k_predict_i = P_k_predict_i + prediction_P_model[i] * self.list_P_k_predict[index][end_i - i]

                # update predicted B, P and add to list
                self.B_k_predict[index] = B_k_predict_i
                self.P_k_predict[index] = P_k_predict_i
                self.list_B_k_predict[index].append(B_k_predict_i)
                self.list_P_k_predict[index].append(P_k_predict_i)

                if len(self.list_B_k_predict[index]) > 2.0 * (self.prediction_sw - 1):
                    del (self.list_B_k_predict[index][0])
                    del (self.list_P_k_predict[index][0])

                ## calculate the probability of model parameter value falls into safe region
                loc = B_k_predict_i
                scale = np.sqrt(P_k_predict_i)
                predict_trigger_indicator = stats.norm.cdf(self.safe_region[index][1], loc, scale) - \
                                            stats.norm.cdf(self.safe_region[index][0], loc, scale)

                if predict_trigger_indicator < self.active_trigger_threshold:
                    predict_active_flag = 1

        active_flag = 0
        if trigger_mode == "prior_trigger":
            active_flag = prior_active_flag
        elif trigger_mode == "posterior_trigger":
            active_flag = predict_active_flag
        elif trigger_mode == "combine_trigger":
            active_flag = prior_active_flag or predict_active_flag

        if self.logging:
            print("prior_active_flag[{}]: {}".format(index, prior_active_flag))
            print("predict_active_flag[{}]: {}".format(index, predict_active_flag))
            print("active_flag[{}]: {}\n".format(index, active_flag))

        return active_flag

    def activate_timing_trigger(self, index):
        # update timer
        self.timer[index] = self.timer[index] + self.supervision_time_interval
        # print("timer: {}".format(self.timer[i]))

        active_flag = 0
        if self.timer[index] >= self.timing_interval:
            active_flag = 1

            # reset timer
            self.timer[index] = 0.0

        return active_flag

    def calculate_safety_AS(self, index, u_i, LIT_i, B_k_prior_i, P_k_prior_i):
        # y(k) = C*B_k*u(k-1) + C*w(k) + v(k)
        # -->
        # u_as_1 = (Y_constraint[0] - LIT_i + v(k) + C*w(k))/(C*B_k)
        # u_as_2 = (Y_constraint[1] - LIT_i - v(k) - C*w(k))/(C*B_k)

        # model parameter
        C_i = self.C[int(index / 2)][int(index / 2)]
        # print("C_i: {}".format(C_i))

        # max deviation model parameter
        if B_k_prior_i > 0:
            max_B_k = B_k_prior_i + self.safety_n_sigma * np.sqrt(P_k_prior_i)
        else:
            max_B_k = B_k_prior_i - self.safety_n_sigma * np.sqrt(P_k_prior_i)
        # print("max_B_k: {}".format(max_B_k))

        # possible max measurement error
        max_uncertainty = self.safety_n_sigma * np.sqrt(self.V[int(index / 2)][int(index / 2)]) + \
                          C_i * self.safety_n_sigma * np.sqrt(self.W[int(index / 2)][int(index / 2)])
        # print("max_uncertainty: {}".format(max_uncertainty))

        # tank water level constraint
        Y_constraint_i = self.Y_constraint[int(index / 2)]
        # print("Y_constraint_i: {}".format(Y_constraint_i))

        # compute safety auxiliary signal
        ## if water level exceed Y_constraint
        if LIT_i < Y_constraint_i[0]:
            LIT_i = Y_constraint_i[0]
        elif LIT_i > Y_constraint_i[1]:
            LIT_i = Y_constraint_i[1]

        ## consider max_B_k>0 or max_B_k<=0
        if max_B_k > 0:
            u_as_1 = (Y_constraint_i[0] - LIT_i + max_uncertainty) / (C_i * max_B_k)
            u_as_2 = (Y_constraint_i[1] - LIT_i - max_uncertainty) / (C_i * max_B_k)
        else:
            u_as_1 = (Y_constraint_i[1] - LIT_i - max_uncertainty) / (C_i * max_B_k)
            u_as_2 = (Y_constraint_i[0] - LIT_i + max_uncertainty) / (C_i * max_B_k)
        # print("u_as_1[{}]: {}".format(index, u_as_1))
        # print("u_as_2[{}]: {}".format(index, u_as_2))

        AS_constraint_low_i = self.U_constraint[index][0] - u_i
        AS_constraint_upper_i = self.U_constraint[index][1] - u_i
        # print("AS_constraint_low_i[{}]: {}".format(index, AS_constraint_low_i))
        # print("AS_constraint_upper_i[{}]: {}".format(index, AS_constraint_upper_i))

        u_safety_low = max(u_as_1, AS_constraint_low_i)
        u_safety_high = min(u_as_2, AS_constraint_upper_i)
        safety_AS_i = [u_safety_low, u_safety_high]
        # print("safety_AS_i: {}".format(safety_AS_i))

        return safety_AS_i

    def calculate_optimal_AS(self, index, B_k_posterior_i):
        # multi-objective optimization:
        ## 1) minimize operational cost of auxiliary signal
        ##          f_C = T*r_C*u'(k)^2
        ## 2) maximize system safety under auxiliary signal
        ##          f_S = r_S*(0.5*(y_u+y_L)-y'(k+1))^2
        ## 3) maximize expected detection improvement
        ##          f_D = r_D*y'(k+1)^2
        ## subject to: 1) u'(k) ∈ [safety_AS_min, safety_AS_max]
        ##             2) y'(k+1) = C*B(k+1)*u'(k)
        ## ->
        ## max(u') f = f_D + f_S - f_C
        ##           = r_D*y'(k+1)^2 + r_S*y'(k+1)^2 - r_S*(y_u+y_L)*y'(k+1) + 0.25*r_S*(y_u+y_L)^2 - T*r_C*u'(k)^2
        ##           = (r_D*C^2*B(k+1)^2 + r_S*C^2*B(k+1)^2 - T*r_C)*u'(k)^2 - r_S*(y_u+y_L)*C*B(k+1)*u'(k+1) + 0.25*r_S*(y_u+y_L)^2
        ## subject to: u'(k) ∈ [safety_AS_min, safety_AS_max]
        ##
        ## convert to the standard form of a QP following CVXOPT
        ## minimize (1/2)*x*P*x + q*x
        ## subject to: G*x < h
        ## -->
        ## min(u') 0.5*(-2.0*r_D*C^2*B(k+1)^2 - 2.0*r_S*C^2*B(k+1)^2 + 2.0*T*r_C)*u'(k)^2 + r_S*(y_u+y_L)*C*B(k+1)*u'(k+1)
        ## subject to: -u'(k) <= safety_AS_min
        ##              u'(k) <= safety_AS_max

        # function parameters
        T = self.T
        r_C_i = self.r_C[index]
        r_S_i = self.r_S[index]
        r_D_i = self.r_D[index]

        # model parameters
        C_i = self.C[int(index / 2)][int(index / 2)]

        # safety auxiliary signal
        safety_AS_min = self.safety_AS[index][0]
        safety_AS_max = self.safety_AS[index][1]
        # print("safety_AS: [{}, {}]".format(safety_AS_min, safety_AS_max))

        # optimization matrices
        ## x = [u'(k)]
        ## P = [-2.0*r_D*C^2*B(k+1)^2 - 2.0*r_S*C^2*B(k+1)^2 + T*r_C]
        P_0_0 = -2.0 * r_D_i * pow(C_i, 2) * pow(B_k_posterior_i, 2) \
                - 2.0 * r_S_i * pow(C_i, 2) * pow(B_k_posterior_i, 2) + T * r_C_i

        P = matrix(np.array([P_0_0]), tc='d')
        # print("P_0_0: {}".format(P_0_0))

        ## x = [u'(k)]
        ## q = [r_S*(y_u+y_L)*C*B(k+1)]
        y_L = self.Y_constraint[int(index / 2)][0]
        y_U = self.Y_constraint[int(index / 2)][1]
        q_0_0 = r_S_i * (y_U + y_L) * C_i * B_k_posterior_i

        q = matrix(np.array([q_0_0]), tc='d')
        # print("q_0_0: {}".format(q_0_0))

        # inequality matrix
        if abs(safety_AS_min) > 0 and abs(safety_AS_max) > 0:
            ## Tips: elements of h to be close to 1.0
            ## x = [u'(k)]
            ## G = [-1.0/safety_AS_min; 1.0/safety_AS_max], h = [1.0; 1.0]
            G = matrix(np.array([[-1.0 / safety_AS_min], [1.0 / safety_AS_max]]), tc='d')
            h = matrix(np.array([[1.0], [1.0]]), tc='d')
        else:
            ## x = [u'(k)]
            ## G = [-1.0; 1.0], h = [safety_AS_min; safety_AS_max]
            G = matrix(np.array([[-1.0], [1.0]]), tc='d')
            h = matrix(np.array([[safety_AS_min], [safety_AS_max]]), tc='d')

        try:
            # cvxopt solver
            ## minimize (1/2)*x*P*x + q*x
            ## subject to: G*x < h
            sol = solvers.qp(P, q, G, h)

        except Exception as err:
            print("Exception: {}".format(err))

            # discrete control parameter
            opt_AS_i = 0.0
            # print("discrete opt_AS_i[{}]: {}\n".format(index, opt_AS_i))

        else:
            # extract optimal value and solution
            status = sol['status']
            opt_AS_i = sol['x'][0]
            costFunction = sol['primal objective']
            # print("status[{}]: {}".format(index, status))
            # print("opt_AS_i[{}]: {}".format(index, opt_AS_i))
            # print("costFunction({}): {}\n".format(index, costFunction))

            # discrete control parameter
            ## time is counted in 0.05s
            if status == "optimal":
                opt_AS_i = round(np.floor(opt_AS_i / 0.05) * 0.05, 2)
            else:
                opt_AS_i = 0.0
            # print("discrete opt_AS_i[{}]: {}\n".format(index, opt_AS_i))

        # handling opt_AS = 0 with active_flag = 1
        if self.active_flag[index] > 0 and opt_AS_i == 0.0:
            # reset auxiliary signal trigger
            self.active_flag[index] = 0

            # reset B_k_prior and P_k_prior
            self.B_k_prior[index][int(index / 2)] = self.B_k_posterior[index][int(index / 2)]
            self.P_k_prior[index][index] = self.P_k_posterior[index][index]

            # reset B_k_predict and P_k_predict
            self.B_k_predict[index][int(index / 2)] = self.B_k_posterior[index][int(index / 2)]
            self.P_k_predict[index][index] = self.P_k_posterior[index][index]

        return opt_AS_i

    def calculate_random_AS(self, index):
        # safety auxiliary signal
        safety_AS_min = self.safety_AS[index][0]
        safety_AS_max = self.safety_AS[index][1]

        # random select auxiliary signal
        random_AS = random.uniform(safety_AS_min, safety_AS_max)

        # discrete control parameter
        ## time is counted in 0.05s
        random_AS = round(np.ceil(random_AS / 0.05) * 0.05, 2)

        return random_AS

    def set_timing_interval(self, timing_interval):
        self.timing_interval = timing_interval

    def set_rnd_seed(self, rnd_seed):
        self.rnd_seed = rnd_seed
        random.seed(self.rnd_seed)

    def get_B_k_prior(self):
        return self.B_k_prior

    def get_P_k_prior(self):
        return self.P_k_prior

    def get_B_k_posterior(self):
        return self.B_k_posterior

    def get_P_k_posterior(self):
        return self.P_k_posterior

    def get_B_k_predict(self):
        return self.B_k_predict

    def get_P_k_predict(self):
        return self.P_k_predict

    def get_active_flag(self):
        return self.active_flag

    def get_opt_AS(self):
        return self.opt_AS
