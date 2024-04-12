class Parameters():
    def __init__(self):
        self.max_pos = 950
        self.params = {}
        etf_params = {
                "edge": .005, # One sided edge from fair
                "fade": .005, # Fade per 100 delta
                "size": 100, # Size of trade
                "edge_slack": .10 # edge to ask for beyond min edge
                }
        contract_params = {
                "edge": .002,
                "fade": .001,
                "size": 100,
                "edge_slack": .10
                }

        for c in ["EPT", "DLO", "MKU", "IGM", "BRV"]:
            self.params[c] = contract_params
        for c in ["SCP", "JAK"]:
            self.params[c] = etf_params