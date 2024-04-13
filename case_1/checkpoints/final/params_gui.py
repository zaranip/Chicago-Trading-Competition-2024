import tkinter as tk
from tkinter import ttk
from params import Parameters

class ParametersGUI:
    def __init__(self, file_path):
        self.file_path = file_path
        self.window = tk.Tk()
        self.window.title("Parameters Editor")
        
        self.load_initial_values()
        self.create_widgets()
        
    def load_initial_values(self):
        with open(self.file_path, "r") as file:
            content = file.read()
            exec(content, globals())
            params = Parameters()
            
            self.max_pos_var = tk.StringVar(value=str(params.max_pos))
            self.min_margin_etf_var = tk.StringVar(value=str(params.etf_params["min_margin"]))
            self.fade_etf_var = tk.StringVar(value=str(params.etf_params["fade"]))
            self.edge_sensitivity_etf_var = tk.StringVar(value=str(params.etf_params["edge_sensitivity"]))
            self.slack_etf_var = tk.StringVar(value=str(params.etf_params["slack"]))
            self.min_margin_contract_var = tk.StringVar(value=str(params.contract_params["min_margin"]))
            self.fade_contract_var = tk.StringVar(value=str(params.contract_params["fade"]))
            self.edge_sensitivity_contract_var = tk.StringVar(value=str(params.contract_params["edge_sensitivity"]))
            self.slack_contract_var = tk.StringVar(value=str(params.contract_params["slack"]))
            self.spreads_var = tk.StringVar(value=str(params.spreads))
            self.level_orders_var = tk.StringVar(value=str(params.level_orders))
            self.etf_margin_var = tk.StringVar(value=str(params.etf_margin))
            self.safety_var = tk.BooleanVar(value=params.safety)
        
    def create_widgets(self):
        max_pos_label = ttk.Label(self.window, text="Max Position:")
        max_pos_label.grid(row=0, column=0, sticky=tk.W)
        max_pos_entry = ttk.Entry(self.window, textvariable=self.max_pos_var)
        max_pos_entry.grid(row=0, column=1)
        self.create_max_pos_buttons()
        
        ttk.Separator(self.window, orient="horizontal").grid(row=1, columnspan=4, sticky="ew")
        
        etf_params_label = ttk.Label(self.window, text="ETF Parameters:")
        etf_params_label.grid(row=2, column=0, sticky=tk.W)
        
        min_margin_etf_label = ttk.Label(self.window, text="Min Margin:")
        min_margin_etf_label.grid(row=3, column=0, sticky=tk.W)
        min_margin_etf_entry = ttk.Entry(self.window, textvariable=self.min_margin_etf_var)
        min_margin_etf_entry.grid(row=3, column=1)
        
        fade_etf_label = ttk.Label(self.window, text="Fade:")
        fade_etf_label.grid(row=4, column=0, sticky=tk.W)
        fade_etf_entry = ttk.Entry(self.window, textvariable=self.fade_etf_var)
        fade_etf_entry.grid(row=4, column=1)
        self.create_fade_etf_buttons()
        
        edge_sensitivity_etf_label = ttk.Label(self.window, text="Edge Sensitivity:")
        edge_sensitivity_etf_label.grid(row=5, column=0, sticky=tk.W)
        edge_sensitivity_etf_entry = ttk.Entry(self.window, textvariable=self.edge_sensitivity_etf_var)
        edge_sensitivity_etf_entry.grid(row=5, column=1)
        self.create_edge_sensitivity_etf_buttons()
        
        slack_etf_label = ttk.Label(self.window, text="Slack:")
        slack_etf_label.grid(row=6, column=0, sticky=tk.W)
        slack_etf_entry = ttk.Entry(self.window, textvariable=self.slack_etf_var)
        slack_etf_entry.grid(row=6, column=1)
        self.create_slack_etf_buttons()
        
        ttk.Separator(self.window, orient="horizontal").grid(row=7, columnspan=4, sticky="ew")
        
        contract_params_label = ttk.Label(self.window, text="Contract Parameters:")
        contract_params_label.grid(row=8, column=0, sticky=tk.W)
        
        min_margin_contract_label = ttk.Label(self.window, text="Min Margin:")
        min_margin_contract_label.grid(row=9, column=0, sticky=tk.W)
        min_margin_contract_entry = ttk.Entry(self.window, textvariable=self.min_margin_contract_var)
        min_margin_contract_entry.grid(row=9, column=1)
        
        fade_contract_label = ttk.Label(self.window, text="Fade:")
        fade_contract_label.grid(row=10, column=0, sticky=tk.W)
        fade_contract_entry = ttk.Entry(self.window, textvariable=self.fade_contract_var)
        fade_contract_entry.grid(row=10, column=1)
        self.create_fade_contract_buttons()
        
        edge_sensitivity_contract_label = ttk.Label(self.window, text="Edge Sensitivity:")
        edge_sensitivity_contract_label.grid(row=11, column=0, sticky=tk.W)
        edge_sensitivity_contract_entry = ttk.Entry(self.window, textvariable=self.edge_sensitivity_contract_var)
        edge_sensitivity_contract_entry.grid(row=11, column=1)
        self.create_edge_sensitivity_contract_buttons()
        
        slack_contract_label = ttk.Label(self.window, text="Slack:")
        slack_contract_label.grid(row=12, column=0, sticky=tk.W)
        slack_contract_entry = ttk.Entry(self.window, textvariable=self.slack_contract_var)
        slack_contract_entry.grid(row=12, column=1)
        self.create_slack_contract_buttons()
        
        ttk.Separator(self.window, orient="horizontal").grid(row=13, columnspan=4, sticky="ew")
        
        spreads_label = ttk.Label(self.window, text="Spreads:")
        spreads_label.grid(row=14, column=0, sticky=tk.W)
        spreads_entry = ttk.Entry(self.window, textvariable=self.spreads_var)
        spreads_entry.grid(row=14, column=1)
        self.create_spreads_buttons()
        
        level_orders_label = ttk.Label(self.window, text="Level Orders:")
        level_orders_label.grid(row=15, column=0, sticky=tk.W)
        level_orders_entry = ttk.Entry(self.window, textvariable=self.level_orders_var)
        level_orders_entry.grid(row=15, column=1)
        self.create_level_orders_buttons()

        etf_margin_label = ttk.Label(self.window, text="ETF Margin:")
        etf_margin_label.grid(row=16, column=0, sticky=tk.W)
        etf_margin_entry = ttk.Entry(self.window, textvariable=self.etf_margin_var)
        etf_margin_entry.grid(row=16, column=1)
        self.create_etf_margin_buttons()
        
        safety_label = ttk.Label(self.window, text="Safety:")
        safety_label.grid(row=17, column=0, sticky=tk.W)
        self.create_safety_buttons()
        
        update_button = ttk.Button(self.window, text="Update", command=self.update_parameters)
        update_button.grid(row=18, columnspan=4)

    def create_max_pos_buttons(self):
        max_pos_values = [50, 100, 150, 200]
        for i, value in enumerate(max_pos_values):
            button = ttk.Button(self.window, text=str(value), command=lambda v=value: self.update_max_pos(v))
            button.grid(row=0, column=i+2)
        
    def create_fade_etf_buttons(self):
        fade_etf_values = [10, 20, 30]
        for i, value in enumerate(fade_etf_values):
            button = ttk.Button(self.window, text=str(value), command=lambda v=value: self.update_fade_etf(v))
            button.grid(row=4, column=i+2)
    
    def create_edge_sensitivity_etf_buttons(self):
        edge_sensitivity_etf_values = [0.5, 1, 1.5]
        for i, value in enumerate(edge_sensitivity_etf_values):
            button = ttk.Button(self.window, text=str(value), command=lambda v=value: self.update_edge_sensitivity_etf(v))
            button.grid(row=5, column=i+2)
    
    def create_slack_etf_buttons(self):
        slack_etf_values = [1, 2, 3]
        for i, value in enumerate(slack_etf_values):
            button = ttk.Button(self.window, text=str(value), command=lambda v=value: self.update_slack_etf(v))
            button.grid(row=6, column=i+2)
    
    def create_fade_contract_buttons(self):
        fade_contract_values = [5, 10, 20, 40, 80]
        for i, value in enumerate(fade_contract_values):
            button = ttk.Button(self.window, text=str(value), command=lambda v=value: self.update_fade_contract(v))
            button.grid(row=10, column=i+2)
    
    def create_edge_sensitivity_contract_buttons(self):
        edge_sensitivity_contract_values = [0.1, 0.25, 0.5, 1]
        for i, value in enumerate(edge_sensitivity_contract_values):
            button = ttk.Button(self.window, text=str(value), command=lambda v=value: self.update_edge_sensitivity_contract(v))
            button.grid(row=11, column=i+2)
    
    def create_slack_contract_buttons(self):
        slack_contract_values = [2, 3, 4]
        for i, value in enumerate(slack_contract_values):
            button = ttk.Button(self.window, text=str(value), command=lambda v=value: self.update_slack_contract(v))
            button.grid(row=12, column=i+2)
    
    def create_spreads_buttons(self):
        spreads_values = [[2, 4, 6], [5, 10, 15], [10, 20, 30]]
        for i, values in enumerate(spreads_values):
            button = ttk.Button(self.window, text=str(values), command=lambda v=values: self.update_spreads(v))
            button.grid(row=14, column=i+2)

    def create_level_orders_buttons(self):
        level_orders_values = [1, 2, 3]
        for i, value in enumerate(level_orders_values):
            button = ttk.Button(self.window, text=str(value), command=lambda v=value: self.update_level_orders(v))
            button.grid(row=15, column=i+2)

    def create_etf_margin_buttons(self):
        etf_margin_values = [60, 80, 100, 120]
        for i, value in enumerate(etf_margin_values):
            button = ttk.Button(self.window, text=str(value), command=lambda v=value: self.update_etf_margin(v))
            button.grid(row=16, column=i+2)

    def create_safety_buttons(self):
        safety_true_button = ttk.Button(self.window, text="True", command=lambda: self.update_safety(True))
        safety_true_button.grid(row=17, column=1)
        safety_false_button = ttk.Button(self.window, text="False", command=lambda: self.update_safety(False))
        safety_false_button.grid(row=17, column=2)

    def update_max_pos(self, value):
        self.max_pos_var.set(str(value))
        self.update_parameters()
    
    def update_fade_etf(self, value):
        self.fade_etf_var.set(str(value))
        self.update_parameters()
    
    def update_edge_sensitivity_etf(self, value):
        self.edge_sensitivity_etf_var.set(str(value))
        self.update_parameters()
    
    def update_slack_etf(self, value):
        self.slack_etf_var.set(str(value))
        self.update_parameters()
    
    def update_fade_contract(self, value):
        self.fade_contract_var.set(str(value))
        self.update_parameters()
    
    def update_edge_sensitivity_contract(self, value):
        self.edge_sensitivity_contract_var.set(str(value))
        self.update_parameters()
    
    def update_slack_contract(self, value):
        self.slack_contract_var.set(str(value))
        self.update_parameters()
    
    def update_spreads(self, values):
        self.spreads_var.set(str(values))
        self.update_parameters()

    def update_level_orders(self, value):
        self.level_orders_var.set(str(value))
        self.update_parameters()

    def update_etf_margin(self, value):
        self.etf_margin_var.set(str(value))
        self.update_parameters()

    def update_safety(self, value):
        self.safety_var.set(value)
        self.update_parameters()
    
    def update_parameters(self):
        max_pos = self.max_pos_var.get()
        min_margin_etf = self.min_margin_etf_var.get()
        fade_etf = self.fade_etf_var.get()
        edge_sensitivity_etf = self.edge_sensitivity_etf_var.get()
        slack_etf = self.slack_etf_var.get()
        min_margin_contract = self.min_margin_contract_var.get()
        fade_contract = self.fade_contract_var.get()
        edge_sensitivity_contract = self.edge_sensitivity_contract_var.get()
        slack_contract = self.slack_contract_var.get()
        spreads = self.spreads_var.get()
        level_orders = self.level_orders_var.get()
        etf_margin = self.etf_margin_var.get()
        safety = self.safety_var.get()
        
        with open(self.file_path, "w") as file:
            file.write(f"""
class Parameters:
    def __init__(self):
        self.max_pos = {max_pos}
        self.params = {{}}
        self.etf_params = {{
            "min_margin": {min_margin_etf},
            "fade": {fade_etf},
            "edge_sensitivity": {edge_sensitivity_etf},
            "slack": {slack_etf}
        }}
        self.contract_params = {{
            "min_margin": {min_margin_contract},
            "fade": {fade_contract},
            "edge_sensitivity": {edge_sensitivity_contract},
            "slack": {slack_contract}
        }}
        for c in ["EPT", "DLO", "MKU", "IGM", "BRV"]:
            self.params[c] = self.contract_params
        for c in ["SCP", "JAK"]:
            self.params[c] = self.etf_params
        self.spreads = {spreads}
        self.level_orders = {level_orders}
        self.etf_margin = {etf_margin}
        self.safety = {safety}

def get_params():
    return Parameters()
""")
        
    def run(self):
        self.window.mainloop()

if __name__ == "__main__":
    file_path = "params.py"
    gui = ParametersGUI(file_path)
    gui.run()