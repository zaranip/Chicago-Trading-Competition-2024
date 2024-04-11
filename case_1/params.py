class Parameters():
    def __init__(self):
        self.max_pos = 950
        self.params = {}
        self.params['EFM'] = {
                "edge": .005, # One sided edge from fair
                "fade": .005, # Fade per 100 delta
                "size": 100, # Size of trade
                "edge_slack": .10 # edge to ask for beyond min edge
                }
        otherparam = {
                "edge": .002,
                "fade": .001,
                "size": 100,
                "edge_slack": .10
                }

        for c in ["EPT", "DLO", "MKU", "IGM", "BRV", "SCP", "JAK"]:
            self.params[c] = otherparam