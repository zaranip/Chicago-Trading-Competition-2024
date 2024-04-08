import subprocess
import sys
import time
import datetime

if __name__ == "__main__":
    counter = 0
    filename = "/Users/divy/Desktop/Chicago-Trading-Competition-2024/case_1/case_1_bot.py"
    while True:
        '''
           Run each round until finish, then wait 10 seconds and run the next round
           for quick copy paste:
              python run.py case_1_bot.py
              python run.py case_1_pipo.py 
        '''

        start_time = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        subprocess.run(["../venv/Scripts/python", filename])
        counter += 1
        with open(f"log/runs/run_{start_time}.txt", "a") as f:
            f.write(f"Finished round {counter} at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")