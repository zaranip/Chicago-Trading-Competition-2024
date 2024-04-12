class Parameters:
    def __init__(self):
        self.max_pos = 950
        self.params = {}
        self.etf_params = {
            "min_margin": 1,
            "fade": 20,
            "edge_sensitivity": 1,
            "slack": 2
        }
        self.contract_params = {
            "min_margin": 1,
            "fade": 20,
            "edge_sensitivity": 0.5,
            "slack": 2
        }
        for c in ["EPT", "DLO", "MKU", "IGM", "BRV"]:
            self.params[c] = self.contract_params
        for c in ["SCP", "JAK"]:
            self.params[c] = self.etf_params

def get_params():
    return Parameters()