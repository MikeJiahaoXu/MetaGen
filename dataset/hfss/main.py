"""
Main process controller for HFSS automation
Monitors and restarts modeling subprocess, handles graceful shutdown
"""
import subprocess
import time
import sys

def start_model_process():
    """Start a new model.py subprocess with face parameter"""
    return subprocess.Popen(['python', 'model.py', '--face', '1'])

if __name__ == '__main__':
    process = start_model_process()
    
    try:
        while True:
            # Check process status every 30 seconds
            if process.poll() is not None:
                print('Model process terminated, restarting...')
                process = start_model_process()
                
            time.sleep(30)
            
    except KeyboardInterrupt:
        print('\nReceived keyboard interrupt, exiting...')
        sys.exit(0)