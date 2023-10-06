# We write plant odes here. The plant would "read" the IO from PLC and decide
# the set of ode functions it follows in the specific current tau ms period.
import math
import numpy as np

from controlSWaT.device.device import *


class Plant:
    def __init__(self, tau, init, noise_mode):
        # time is counted in tau seconds
        self.tau = tau

        # tank level
        self.noisy_h_t101 = self.h_t101 = init[0]  # Raw Water Tank:     T101
        self.noisy_h_t301 = self.h_t301 = init[1]  # UF Feed-water Tank: T301
        self.noisy_h_t401 = self.h_t401 = init[2]  # RO Feed-water Tank: T401
        self.noisy_h_t601 = self.h_t601 = init[3]  # RO Permeate Tank:   T601
        self.noisy_h_t602 = self.h_t602 = init[4]  # UF Backwash Tank:   T602

        # key parameters
        ## P1: raw water
        self.f_mv101 = self.drift_f_mv101 = 2.3 * 1000000000 / 3600  # (m^3/s) Raw Water Inlet On/Off Valve
        self.S_t101 = 1.5 * 1000000
        self.f_p101 = 2.0 * 1000000000 / 3600  # (m^3/s) Raw Water Transfer Pump
        self.noisy_drift_B_0 = self.drift_B_0 = self.noisy_B_0 = self.B_0 = self.f_mv101 / self.S_t101
        self.noisy_drift_B_1 = self.drift_B_1 = self.noisy_B_1 = self.B_1 = - self.f_p101 / self.S_t101

        ## P2: chemical dosing
        self.f_mv201 = 2.0 * 1000000000 / 3600  # (m^3/s) Raw Water Tank Outlet On/Off Valve

        ## P3: ultrafiltration
        self.S_t301 = 1.5 * 1000000
        self.f_p301 = 2.0 * 1000000000 / 3600  # (m^3/s) UF Feed-water Pump
        self.f_mv302 = 2.0 * 1000000000 / 3600  # (m^3/s) UF On/Off Valve
        self.noisy_drift_B_2 = self.drift_B_2 = self.noisy_B_2 = self.B_2 = self.f_mv201 / self.S_t301
        self.noisy_drift_B_3 = self.drift_B_3 = self.noisy_B_3 = self.B_3 = - self.f_p301 / self.S_t301

        ## P4: dechlorination
        self.S_t401 = 1.5 * 1000000
        self.f_p401 = 2.0 * 1000000000 / 36001  # (m^3/s) RO Feed-water Pump
        self.noisy_drift_B_4 = self.drift_B_4 = self.noisy_B_4 = self.B_4 = self.f_mv302 / self.S_t401

        ## P5: RO
        self.S_t501 = 1.5 * 1000000
        self.f_p501 = 0  # (m^3/s) RO Pump (do not active)
        self.f_mv501 = 2.0 * 1000000000 / 3600  # (m^3/s) RO Permeate Valve
        self.f_mv502 = 0.00006111  # (m^3/s) Backwash Valve
        self.f_mv503 = 0.00049  # (m^3/s) Permeate Reject Valve

        ## P6-1: RO Permeate
        self.S_t601 = 1.5 * 1000000
        self.f_p601 = 2.0 * 1000000000 / 36001  # (m^3/s) RO Permeate Transfer Pump
        self.LIT601_AL = 200
        self.LIT601_AH = 700

        ## P6-2: UF Backwash
        self.S_t602 = 1.5 * 1000000
        self.f_p602 = 2.0 * 1000000000 / 3600  # (m^3/s) UF Backwash Tank Pump
        self.LIT602_AL = 200
        self.LIT602_AH = 700

        # process noise and sensor noise: numpy.random.normal(loc=0.0, scale=1.0, size=None) scale is standard deviation
        # process noise: 0.01*1000000000/3600/(1.5*1000000)=0.002
        if noise_mode == 1:
            self.process_noise_sigma = 0.002 * 5
            self.sensor_noise_sigma = 0.02
        else:
            self.process_noise_sigma = 0.0
            self.sensor_noise_sigma = 0.0

        # longterm model deviation: slow drift
        ## B_t = B_offset_percent * B_0 * sin((2*pi/period) * time) + B_0
        self.trigger_B_drift = 0  # 0~4
        self.deviation_B_index = 0
        self.B_offset_percent = 0.0
        self.period = 6 * 30 * 60 # 6*1800 seconds

    def actuator(self, P1, P2, P3, P4, P5, P6, safe_control_mode):
        # Viewer should be notified that passing the output value from plc directly
        # to the input value(return value) is not quite true. For actuators
        # (in here, we are talking about motor valve, pump, pressure pump and ultra violet),
        # the behavior is unknown, it could take seconds for the actuators
        # to process the instruction, meaning there's a lapse
        # between the instruction sent and carried out.

        if safe_control_mode:
            self.trigger_positive_trace = 0
            self.hit_B_index = 0
            self.hit_B_coefficient = 1.0

            # tank101
            if self.noisy_h_t101 > 1100:
                P1.MV101.DI_ZSO = 0
                P1.MV101.DI_ZSC = 1

            elif self.noisy_h_t101 < 250:
                P1.P101.DI_Run = 0
                P1.P102.DI_Run = 0

            # tank301
            if self.noisy_h_t301 > 1200:
                P2.MV201.DI_ZSO = 0
                P2.MV201.DI_ZSC = 1

                P1.P101.DI_Run = 0
                P1.P102.DI_Run = 0

            elif self.noisy_h_t301 < 250:
                P3.P301.DI_Run = 0
                P3.P302.DI_Run = 0

            # tank401
            if self.noisy_h_t401 > 1200:
                P3.P301.DI_Run = 0
                P3.P302.DI_Run = 0

                P3.MV302.DI_ZSO = 0
                P3.MV302.DI_ZSC = 1

            elif self.noisy_h_t401 < 250:
                P4.P401.DI_Run = 0
                P4.P402.DI_Run = 0

        else:
            P1.MV101.DI_ZSO = P1.MV101.DO_Open  # Raw Water Inlet On/Off Valve
            P1.MV101.DI_ZSC = P1.MV101.DO_Close
            P2.MV201.DI_ZSO = P2.MV201.DO_Open  # Raw Water Tank Outlet On/Off Valve
            P2.MV201.DI_ZSC = P2.MV201.DO_Close
            P3.MV301.DI_ZSO = P3.MV301.DO_Open  # Bash wash On/Off Valve
            P3.MV301.DI_ZSC = P3.MV301.DO_Close
            P3.MV302.DI_ZSO = P3.MV302.DO_Open
            P3.MV302.DI_ZSC = P3.MV302.DO_Close
            P3.MV303.DI_ZSO = P3.MV303.DO_Open
            P3.MV303.DI_ZSC = P3.MV303.DO_Close
            P3.MV304.DI_ZSO = P3.MV304.DO_Open
            P3.MV304.DI_ZSC = P3.MV304.DO_Close
            P5.MV501.DI_ZSO = P5.MV501.DO_Open  # RO Permeate Valve
            P5.MV501.DI_ZSC = P5.MV501.DO_Close
            P5.MV502.DI_ZSO = P5.MV502.DO_Open  # Backwash Valve
            P5.MV502.DI_ZSC = P5.MV502.DO_Close
            P5.MV503.DI_ZSO = P5.MV503.DO_Open  # Permeate Reject Valve
            P5.MV503.DI_ZSC = P5.MV503.DO_Close
            P5.MV504.DI_ZSO = P5.MV504.DO_Open
            P5.MV504.DI_ZSC = P5.MV504.DO_Close

            P1.P101.DI_Run = P1.P101.DO_Start
            P1.P102.DI_Run = P1.P102.DO_Start
            P3.P301.DI_Run = P3.P301.DO_Start
            P3.P302.DI_Run = P3.P302.DO_Start
            P4.P401.DI_Run = P4.P401.DO_Start
            P4.P402.DI_Run = P4.P402.DO_Start
            P5.P501.DI_Run = (P5.P501_VSD_Out.Start or not P5.P501_VSD_Out.Stop) + 0
            P5.P502.DI_Run = (P5.P502_VSD_Out.Start or not P5.P502_VSD_Out.Stop) + 0
            P6.P601.DI_Run = P6.P601.DO_Start
            P6.P602.DI_Run = P6.P602.DO_Start

            P4.UV401.DI_Run = P4.UV401.DO_Start

    def plant(self, P1, P2, P3, P4, P5, P6, HMI, time):
        # feeding water to tank101 (open MV101)
        if P1.MV101.DI_ZSO == 1:
            if self.trigger_B_drift == 1 and self.deviation_B_index == 0:
                self.drift_B_0 = self.B_offset_percent * self.B_0 * \
                                 math.sin((2 * math.pi / self.period) * time) + self.B_0

                self.noisy_drift_B_0 = self.drift_B_0 + np.random.normal(0.0, self.process_noise_sigma)
                self.h_t101 = self.h_t101 + self.noisy_drift_B_0 * self.tau
            else:
                self.noisy_B_0 = self.B_0 + np.random.normal(0.0, self.process_noise_sigma)
                self.h_t101 = self.h_t101 + self.noisy_B_0 * self.tau

        # drawing water from tank101 (start P101 or P102)
        if P1.P101.DI_Run == 1:
            if self.trigger_B_drift == 1 and self.deviation_B_index == 1:
                self.drift_B_1 = self.B_offset_percent * self.B_1 * \
                                 math.sin((2 * math.pi / self.period) * time) + self.B_1

                self.noisy_drift_B_1 = self.drift_B_1 + np.random.normal(0.0, self.process_noise_sigma)
                self.h_t101 = self.h_t101 + self.noisy_drift_B_1 * self.tau
            else:
                self.noisy_B_1 = self.B_1 + np.random.normal(0.0, self.process_noise_sigma)
                self.h_t101 = self.h_t101 + self.noisy_B_1 * self.tau

        if P1.P102.DI_Run == 1:
            if self.trigger_B_drift == 1 and self.deviation_B_index == 1:
                self.drift_B_1 = self.B_offset_percent * self.B_1 * \
                                 math.sin((2 * math.pi / self.period) * time) + self.B_1

                self.noisy_drift_B_1 = self.drift_B_1 + np.random.normal(0.0, self.process_noise_sigma)
                self.h_t101 = self.h_t101 + self.noisy_drift_B_1 * self.tau
            else:
                self.noisy_B_1 = self.B_1 + np.random.normal(0.0, self.process_noise_sigma)
                self.h_t101 = self.h_t101 + self.noisy_B_1 * self.tau

        # feeding water to tank301 (open MV201 and start P101/2)
        if P2.MV201.DI_ZSO == 1 and (P1.P101.DI_Run == 1 or P1.P101.DI_Run == 1):
            if self.trigger_B_drift == 1 and self.deviation_B_index == 2:
                self.drift_B_2 = self.B_offset_percent * self.B_2 * \
                                 math.sin((2 * math.pi / self.period) * time) + self.B_2

                self.noisy_drift_B_2 = self.drift_B_2 + np.random.normal(0.0, self.process_noise_sigma)
                self.h_t301 = self.h_t301 + self.noisy_drift_B_2 * self.tau
            else:
                self.noisy_B_2 = self.B_2 + np.random.normal(0.0, self.process_noise_sigma)
                self.h_t301 = self.h_t301 + self.noisy_B_2 * self.tau

        # drawing water from tank301 (start P301 or P302)
        if P3.P301.DI_Run == 1:
            if self.trigger_B_drift == 1 and self.deviation_B_index == 3:
                self.drift_B_3 = self.B_offset_percent * self.B_3 * \
                                 math.sin((2 * math.pi / self.period) * time) + self.B_3

                self.noisy_drift_B_3 = self.drift_B_3 + np.random.normal(0.0, self.process_noise_sigma)
                self.h_t301 = self.h_t301 + self.noisy_drift_B_3 * self.tau
            else:
                self.noisy_B_3 = self.B_3 + np.random.normal(0.0, self.process_noise_sigma)
                self.h_t301 = self.h_t301 + self.noisy_B_3 * self.tau

        if P3.P302.DI_Run == 1:
            if self.trigger_B_drift == 1 and self.deviation_B_index == 3:
                self.drift_B_3 = self.B_offset_percent * self.B_3 * \
                                 math.sin((2 * math.pi / self.period) * time) + self.B_3

                self.noisy_drift_B_3 = self.drift_B_3 + np.random.normal(0.0, self.process_noise_sigma)
                self.h_t301 = self.h_t301 + self.noisy_drift_B_3 * self.tau
            else:
                self.noisy_B_3 = self.B_3 + np.random.normal(0.0, self.process_noise_sigma)
                self.h_t301 = self.h_t301 + self.noisy_B_3 * self.tau

        # UF flushing procedure, 30 sec (start P301 or P302, open MV304 and stop P602)
        if (P3.P301.DI_Run == 1 or P3.P302.DI_Run == 1) and \
                P3.MV301.DI_ZSC == 1 and P3.MV302.DI_ZSC == 1 and P3.MV303.DI_ZSC == 1 and P3.MV304.DI_ZSO == 1 and \
                P6.P602.DI_Run == 0:
            pass

        # UF ultra filtration procedure, 30 min (start P301 or P302, open MV302 and stop P602)
        if (P3.P301.DI_Run == 1 or P3.P302.DI_Run == 1) and \
                P3.MV301.DI_ZSC == 1 and P3.MV302.DI_ZSO == 1 and P3.MV303.DI_ZSC == 1 and P3.MV304.DI_ZSC == 1 and \
                P6.P602.DI_Run == 0:
            if self.trigger_B_drift == 1 and self.deviation_B_index == 4:
                self.drift_B_4 = self.B_offset_percent * self.B_4 * \
                                 math.sin((2 * math.pi / self.period) * time) + self.B_4

                self.noisy_drift_B_4 = self.drift_B_4 + np.random.normal(0.0, self.process_noise_sigma)
                self.h_t401 = self.h_t401 + self.noisy_drift_B_4 * self.tau
            else:
                self.noisy_B_4 = self.B_4 + np.random.normal(0.0, self.process_noise_sigma)
                self.h_t401 = self.h_t401 + self.noisy_B_4 * self.tau

        # UF back wash procedure, 45 sec (stop P301 and P302, open MV301 and MV303, and start P602)
        if P3.P301.DI_Run == 0 and P3.P302.DI_Run == 0 and \
                P3.MV301.DI_ZSO == 1 and P3.MV302.DI_ZSC == 1 and P3.MV303.DI_ZSO == 1 and P3.MV304.DI_ZSC == 1 and \
                P6.P602.DI_Run == 1:
            self.h_t602 = self.h_t602 - \
                          (self.f_p602 / self.S_t602 + np.random.normal(0.0, self.process_noise_sigma)) * self.tau

        # UF feed tank draining procedure, 1 min (stop P301 and P302, open MV303 and MV304, and stop P602)
        if P3.P301.DI_Run == 0 and P3.P302.DI_Run == 0 and \
                P3.MV301.DI_ZSC == 1 and P3.MV302.DI_ZSC == 1 and P3.MV303.DI_ZSO == 1 and P3.MV304.DI_ZSO == 1 and \
                P6.P602.DI_Run == 0:
            pass

        # pumping water from t401 (start P401 or P402)
        if P4.P401.DI_Run == 1:
            self.h_t401 = self.h_t401 - \
                          (self.f_p401 / self.S_t401 + np.random.normal(0.0, self.process_noise_sigma)) * self.tau

        if P4.P402.DI_Run == 1:
            self.h_t401 = self.h_t401 - \
                          (self.f_p401 / self.S_t401 + np.random.normal(0.0, self.process_noise_sigma)) * self.tau

        # procedure for RO normal functioning with product of permeate 60% and backwash 40%
        # (start P401 or P402, start P501 or P502, open MV501 and MV502)
        if (P4.P401.DI_Run == 1 or P4.P402.DI_Run == 1) and \
                (P5.P501.DI_Run == 1 or P5.P502.DI_Run == 1) and \
                P5.MV501.DI_ZSO == 1 and P5.MV502.DI_ZSO == 1 and P5.MV503.DI_ZSC == 1 and P5.MV504.DI_ZSC == 1:
            self.h_t601 = self.h_t601 + \
                          (self.f_mv501 / self.S_t601 + np.random.normal(0.0, self.process_noise_sigma)) * self.tau
            self.h_t602 = self.h_t602 + \
                          (self.f_mv502 / self.S_t602 + np.random.normal(0.0, self.process_noise_sigma)) * self.tau

        # procedure for RO flushing with product of backwash 60% and drain 40%
        # (start P401 or P402, start P501 or P502, open MV503 and MV504)
        elif (P4.P401.DI_Run == 1 or P4.P402.DI_Run == 1) and \
                (P5.P501.DI_Run == 1 or P5.P502.DI_Run == 1) and \
                P5.MV501.DI_ZSC == 1 and P5.MV502.DI_ZSC == 1 and P5.MV503.DI_ZSO == 1 and P5.MV504.DI_ZSO == 1:
            self.h_t602 = self.h_t602 + \
                          (self.f_mv503 / self.S_t602 + np.random.normal(0.0, self.process_noise_sigma)) * self.tau

        # pumping water out of tank601 and feeding water to tank101 (start P601)
        if P6.P601.DI_Run == 1:
            self.h_t601 = self.h_t601 - \
                          (self.f_p601 / self.S_t601 + np.random.normal(0.0, self.process_noise_sigma)) * self.tau
            self.h_t101 = self.h_t101 + \
                          (self.f_p601 / self.S_t101 + np.random.normal(0.0, self.process_noise_sigma)) * self.tau

        # sensor noise
        self.noisy_h_t101 = min(max(self.h_t101 + np.random.normal(0.0, self.sensor_noise_sigma), 0.0), 1350.0)
        self.noisy_h_t301 = min(max(self.h_t301 + np.random.normal(0.0, self.sensor_noise_sigma), 0.0), 1350.0)
        self.noisy_h_t401 = min(max(self.h_t401 + np.random.normal(0.0, self.sensor_noise_sigma), 0.0), 1350.0)
        self.noisy_h_t601 = min(max(self.h_t601 + np.random.normal(0.0, self.sensor_noise_sigma), 0.0), 1350.0)
        self.noisy_h_t602 = min(max(self.h_t602 + np.random.normal(0.0, self.sensor_noise_sigma), 0.0), 1350.0)

        # update HMI
        HMI.LIT101.Pv = self.noisy_h_t101
        HMI.LIT101.set_alarm()
        HMI.LIT301.Pv = self.noisy_h_t301
        HMI.LIT301.set_alarm()
        HMI.LIT401.Pv = self.noisy_h_t401
        HMI.LIT401.set_alarm()

        if self.noisy_h_t601 > self.LIT601_AH:
            HMI.LSH601.Alarm = True
        if self.noisy_h_t601 < self.LIT601_AL:
            HMI.LSL601.Alarm = True

        if self.noisy_h_t602 > self.LIT602_AH:
            HMI.LSH602.Alarm = True
        if self.noisy_h_t602 < self.LIT602_AL:
            HMI.LSL602.Alarm = True

    def set_longterm_B_drift(self, trigger_B_drift, deviation_B_index, B_offset_percent, period):
        # longterm model deviation: slow drift
        ## B_t = B_offset_percent * B_0 * sin((2*pi/period) * loop) + B_0
        self.trigger_B_drift = trigger_B_drift
        self.deviation_B_index = deviation_B_index
        self.B_offset_percent = B_offset_percent
        self.period = period

    def get_drift_B(self, B_index):
        if B_index == 0:
            return self.drift_B_0

        elif B_index == 1:
            return self.drift_B_1

        elif B_index == 2:
            return self.drift_B_2

        elif B_index == 3:
            return self.drift_B_3

        elif B_index == 4:
            return self.drift_B_4
