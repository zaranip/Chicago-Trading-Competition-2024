
class Parameters:
    def __init__(self):
        self.max_pos = 200
        self.params = {}
        self.etf_params = {
            "min_margin": 1,
            "fade": 20,
            "edge_sensitivity": 1,
            "slack": 2
        }
        self.contract_params = {
            "min_margin": 1,
            "fade": 80,
            "edge_sensitivity": 1,
            "slack": 2
        }
        for c in ["EPT", "DLO", "MKU", "IGM", "BRV"]:
            self.params[c] = self.contract_params
        for c in ["SCP", "JAK"]:
            self.params[c] = self.etf_params
        self.spreads = [5, 10, 15]
        self.level_orders = 2

def get_params():
    return Parameters()
