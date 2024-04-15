
class Parameters:
    def __init__(self):
        self.max_pos = 200
        self.params = {}
        self.etf_params = {
            "min_margin": 1,
            "fade": 20,
            "edge_sensitivity": 0.5,
            "slack": 1
        }
        self.contract_params = {
            "min_margin": 1,
            "fade": 20,
            "edge_sensitivity": 0.1,
            "slack": 4
        }
        for c in ["EPT", "DLO", "MKU", "IGM", "BRV"]:
            self.params[c] = self.contract_params
        for c in ["SCP", "JAK"]:
            self.params[c] = self.etf_params
        self.spreads = [10, 20, 30]
        self.level_orders = 2
        self.etf_margin = 60
        self.safety = False

def get_params():
    return Parameters()
