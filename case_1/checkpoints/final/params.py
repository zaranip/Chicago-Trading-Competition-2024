
class Parameters:
    def __init__(self):
        self.max_pos = 200
        self.params = {}
        self.etf_params = {
            "min_margin": 1,
            "fade": 10,
            "edge_sensitivity": 1.5,
            "slack": 3
        }
        self.contract_params = {
            "min_margin": 1,
            "fade": 80,
            "edge_sensitivity": 0.25,
            "slack": 4
        }
        for c in ["EPT", "DLO", "MKU", "IGM", "BRV"]:
            self.params[c] = self.contract_params
        for c in ["SCP", "JAK"]:
            self.params[c] = self.etf_params
        self.spreads = [2, 4, 6]
        self.level_orders = 3
        self.etf_margin = 120
        self.safety = False

def get_params():
    return Parameters()
