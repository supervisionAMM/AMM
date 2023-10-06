import os
import numpy as np

from controlSWaT.IO import *
from controlSWaT.SCADA import H
from controlSWaT.plc import plc1, plc2, plc3, plc4, plc5, plc6
from controlSWaT.plant.plant import Plant

from supervision.detectors.AMM import AMM
from supervision.safeguard.switcher import Switcher


def main():
    # seed
    np.random.seed(1)

    # general configurations
    logging_mode = True      # whether output the process info
    supervision_mode = True  # whether activate AMM and switch to mandatory controller
    noise_mode = 1
    safe_control_mode = 0
    tau = 0.05  # 50ms

    # time is counted in tau seconds, (1.0/tau)*x*y, x unit seconds, y unit minutes
    maxstep = int(1.0 / tau) * 60 * 30

    print('initializing plant\n')
    init = [498.5885, 827.4722, 875.6601, 351.1663, 273.3779]
    print('init: {}\n'.format(init))

    # initializing plant
    SWaT_Plant = Plant(tau, init, noise_mode)

    print('defining I/Os\n')
    IO_DI_WIFI = DI_WIFI()  # whether PLC processes wireless signal
    IO_P1 = P1()  # define I/Os for sensors and actuators of each process
    IO_P2 = P2()
    IO_P3 = P3()
    IO_P4 = P4()
    IO_P5 = P5()
    IO_P6 = P6()

    print('initializing SCADA HMI\n')
    HMI = H()

    print('initializing PLCs\n')
    PLC1 = plc1.PLC1(HMI)
    PLC2 = plc2.PLC2(HMI)
    PLC3 = plc3.PLC3(HMI)
    PLC4 = plc4.PLC4(HMI)
    PLC5 = plc5.PLC5(HMI)
    PLC6 = plc6.PLC6(HMI)

    # initializing supervision
    if supervision_mode:
        print('initializing supervision\n')

        ## supervision time interval
        d = 1.0
        interval_num = int(d / tau)

        if logging_mode:
            print('d: {}s, interval_num: {}\n'.format(d, interval_num))

        ## supervision modules
        detector = AMM(False, 'combine_trigger', 'optimal_design')
        switcher = Switcher()

        ## supervision control signals and water levels
        U_101 = np.zeros(2)
        U_301 = np.zeros(2)
        U_401 = np.zeros(2)
        list_YS = []

    # tracing
    if not os.path.exists("controlSWaT/trace"):
        os.makedirs("controlSWaT/trace")

    if supervision_mode:
        trace_path = 'controlSWaT/trace/swat_AMM_trace-init-' + str(init) + '.txt'
    else:
        trace_path = 'controlSWaT/trace/swat_PI_trace-init-' + str(init) + '.txt'
    trace_fw = open(trace_path, 'w+')

    # main loop body
    for step in range(0, maxstep):
        # second, minute and hour pulses
        Sec_P = not bool(step % int(1.0 / tau))
        Min_P = not bool(step % (int(1.0 / tau) * 60))

        # set safe control mode
        if supervision_mode:
            safe_control_mode = switcher.get_switch_mode()

        # solving out plant odes in 50 ms
        time = step * 0.05
        SWaT_Plant.actuator(IO_P1, IO_P2, IO_P3, IO_P4, IO_P5, IO_P6, safe_control_mode)
        SWaT_Plant.plant(IO_P1, IO_P2, IO_P3, IO_P4, IO_P5, IO_P6, HMI, time)

        if not supervision_mode:  # identification or control
            # tracing tank levels y(k)
            trace_fw.write(str(HMI.LIT101.Pv) + ',' + str(HMI.LIT301.Pv) + ',' + str(HMI.LIT401.Pv) + ',')
            trace_fw.write(str(SWaT_Plant.noisy_h_t601) + ',' + str(SWaT_Plant.noisy_h_t602) + '\n')

        # supervision process
        if supervision_mode:
            ## measured water levels
            if switcher.get_switch_mode() == 0 and step % interval_num == 0:
                list_YS.append([HMI.LIT101.Pv, HMI.LIT301.Pv, HMI.LIT401.Pv])

            ## model deviation detection
            if switcher.get_switch_mode() == 0:
                if int(step / interval_num) >= 1 and step % interval_num == 0:
                    U = np.array([[U_101[0], 0.0, 0.0],
                                  [U_101[1], 0.0, 0.0],
                                  [0.0, U_301[0], 0.0],
                                  [0.0, U_301[1], 0.0],
                                  [0.0, 0.0, U_401[0]],
                                  [0.0, 0.0, U_401[1]]])

                    Y = np.array([[list_YS[len(list_YS) - 1][0] - list_YS[len(list_YS) - 2][0], 0.0, 0.0],
                                  [0.0, list_YS[len(list_YS) - 1][1] - list_YS[len(list_YS) - 2][1], 0.0],
                                  [0.0, 0.0, list_YS[len(list_YS) - 1][2] - list_YS[len(list_YS) - 2][2]]])

                    LIT = np.array([list_YS[len(list_YS) - 1][0],
                                    list_YS[len(list_YS) - 1][1],
                                    list_YS[len(list_YS) - 1][2]])

                    ## u(k-1), y(k)
                    alarm, active_flag, opt_AS = detector.deviation_detector(U, Y, LIT)
                    B_k_posterior = detector.get_B_k_posterior()
                    P_k_posterior = detector.get_P_k_posterior()

                    ## tracing supervision
                    outline_u = str(U[0][0]) + ',' + str(U[1][0]) + ',' + \
                                str(U[2][1]) + ',' + str(U[3][1]) + ',' + \
                                str(U[4][2]) + ',' + str(U[5][2])
                    trace_fw.write(outline_u + '\n')

                    outline_y = str(Y[0][0]) + ',' + str(Y[1][1]) + ',' + str(Y[2][2])
                    trace_fw.write(outline_y + '\n')

                    outline_B = str(B_k_posterior[0][0]) + ',' + str(B_k_posterior[1][0]) + ',' + \
                                str(B_k_posterior[2][1]) + ',' + str(B_k_posterior[3][1]) + ',' + \
                                str(B_k_posterior[4][2]) + ',' + str(B_k_posterior[5][2])
                    trace_fw.write(outline_B + '\n')

                    outline_Var = str(P_k_posterior[0][0]) + ',' + str(P_k_posterior[1][1]) + ',' + \
                                  str(P_k_posterior[2][2]) + ',' + str(P_k_posterior[3][3]) + ',' + \
                                  str(P_k_posterior[4][4]) + ',' + str(P_k_posterior[5][5])
                    trace_fw.write(outline_Var + '\n')
                    trace_fw.flush()

                    for i in range(0, len(alarm)):
                        if alarm[i] == 1:
                            switcher.set_switch_mode(alarm[i])
                            alarm_index = step
                            print('ending simulation:')
                            print('alarm at index: {}s for B_{}'.format(alarm_index, i))
                            print('B_k_posterior:\n{}'.format(B_k_posterior))
                            print('P_k_posterior:\n{}\n'.format(P_k_posterior))
                            break

        # PLC working
        PLC1.Pre_Main_Raw_Water(IO_P1, HMI)
        PLC2.Pre_Main_UF_Feed_Dosing(IO_P2, HMI)
        PLC3.Pre_Main_UF_Feed(IO_P3, HMI, Sec_P, Min_P)
        PLC4.Pre_Main_RO_Feed_Dosing(IO_P4, HMI)
        PLC5.Pre_Main_High_Pressure_RO(IO_P5, HMI, Sec_P, Min_P)
        PLC6.Pre_Main_Product(IO_P6, HMI)

        if not supervision_mode:
            # tracing MV and pump control signal u(k)
            trace_fw.write(str(IO_P1.MV101.DO_Open) + ',' + str(IO_P2.MV201.DO_Open) + ',' +
                           str(IO_P3.MV301.DO_Open) + ',' + str(IO_P3.MV302.DO_Open) + ',' +
                           str(IO_P3.MV303.DO_Open) + ',' + str(IO_P3.MV304.DO_Open) + ',' +
                           str(IO_P5.MV501.DO_Open) + ',' + str(IO_P5.MV502.DO_Open) + ',' +
                           str(IO_P5.MV503.DO_Open) + ',' + str(IO_P5.MV504.DO_Open) + '\n')
            trace_fw.write(str(IO_P1.MV101.DO_Close) + ',' + str(IO_P2.MV201.DO_Close) + ',' +
                           str(IO_P3.MV301.DO_Close) + ',' + str(IO_P3.MV302.DO_Close) + ',' +
                           str(IO_P3.MV303.DO_Close) + ',' + str(IO_P3.MV304.DO_Close) + ',' +
                           str(IO_P5.MV501.DO_Close) + ',' + str(IO_P5.MV502.DO_Close) + ',' +
                           str(IO_P5.MV503.DO_Close) + ',' + str(IO_P5.MV504.DO_Close) + '\n')
            trace_fw.write(str(IO_P1.P101.DO_Start) + ',' + str(IO_P1.P102.DO_Start) + ',' +
                           str(IO_P3.P301.DO_Start) + ',' + str(IO_P3.P302.DO_Start) + ',' +
                           str(IO_P4.P401.DO_Start) + ',' + str(IO_P4.P402.DO_Start) + ',' +
                           str(IO_P5.P501_VSD_Out.Start) + ',' + str(IO_P5.P502_VSD_Out.Start) + ',' +
                           str(IO_P6.P601.DO_Start) + ',' + str(IO_P6.P602.DO_Start) + '\n')

        # supervision control signal
        if supervision_mode:
            if switcher.get_switch_mode() == 0:
                if step % interval_num == 0:
                    U_101 = np.zeros(2)
                    U_301 = np.zeros(2)
                    U_401 = np.zeros(2)

                # U_101 = [sum(MV101), sum(P101||P102)]
                if IO_P1.MV101.DO_Open == 1:
                    U_101[0] = round(U_101[0] + tau, 3)
                if IO_P1.P101.DO_Start == 1 or IO_P1.P102.DO_Start == 1:
                    U_101[1] = round(U_101[1] + tau, 3)

                # U_301 = [sum(MV201&&(P101||P102)), sum(P301||P302)]
                if IO_P2.MV201.DO_Open == 1 and (IO_P1.P101.DO_Start == 1 or IO_P1.P102.DO_Start == 1):
                    U_301[0] = round(U_301[0] + tau, 3)
                if IO_P3.P301.DO_Start == 1 or IO_P3.P302.DO_Start == 1:
                    U_301[1] = round(U_301[1] + tau, 3)

                # U_401 = [sum((P301||P302)&&~MV301&&MV302&&~MV303&&~MV304&&~P602),
                #          sum(P401||P402)]
                if (IO_P3.P301.DO_Start == 1 or IO_P3.P302.DO_Start == 1) and \
                        IO_P3.MV301.DO_Open == 0 and IO_P3.MV302.DO_Open == 1 and IO_P3.MV303.DO_Open == 0 and IO_P3.MV304.DO_Open == 0 and \
                        IO_P6.P602.DO_Start == 0:
                    U_401[0] = round(U_401[0] + tau, 3)
                if IO_P4.P401.DO_Start == 1 or IO_P4.P402.DO_Start == 1:
                    U_401[1] = round(U_401[1] + tau, 3)

    trace_fw.close()


if __name__ == '__main__':
    main()