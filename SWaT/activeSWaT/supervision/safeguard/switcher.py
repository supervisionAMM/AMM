class Switcher:

    def __init__(self):
        # switcher parameter
        self.switch_mode = 0

    def set_switch_mode(self, alarm):
        if alarm == 1:
            self.switch_mode = 1

    def get_switch_mode(self):
        return self.switch_mode
