import subprocess
import sys
import time
import datetime

if __name__ == "__main__":
    counter = 0
    filename = sys.argv[1]
    while True:
        '''
           Run each round until finish, then wait 10 seconds and run the next round
           for quick copy paste:
              python run.py case_1_bot.py
              python run.py case_1_pipo.py 
        '''


        subprocess.run(["python", filename])
        with open("log/runs.txt", "a") as f:
            f.write(f"Finished round {counter + 1} at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        time.sleep(5)