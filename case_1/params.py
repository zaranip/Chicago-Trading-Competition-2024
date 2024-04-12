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
            "fade": 20, #increasing fade reduces profit AND loss (hitter); decreasing fade increases profit AND loss (non-hitter)
            "edge_sensitivity": 1, #increasing to 1 increases frequency; decreasing to 0.1 increases profit per trade; switch when expected pnl is going down
            "slack":2    #2: 1-3; 3: 1-4; 4: 1-5
        }

        self.spreads = [5, 10, 20]
        self.level_orders = 2
        self.etf_margin = 55

def get_params():
    return Parameters()