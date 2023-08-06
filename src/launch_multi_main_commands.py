import sys
import subprocess as sp
from multiprocessing import Process
import time


def run_command(command):
    sp.run(command, shell=True)



if __name__ == "__main__":
    
    commands = sys.argv[1].split(';')
    processes = []

    for cmd in commands:
        if cmd.strip():  # Ignore empty commands
            cmd = cmd.strip()
            print("launching:", cmd)
            p = Process(target=run_command, args=(cmd,))
            p.start()
            processes.append(p)
            time.sleep(120)

    # Wait for all processes to finish
    for p in processes:
        p.join()