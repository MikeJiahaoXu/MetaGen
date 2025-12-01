"""
HFSS Model Generator
Creates parameterized HFSS models from image dataset using multi-threading
"""
import os
import random
import argparse
import threading
import time
import win32com.client
import pythoncom
import numpy as np
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from utils import new_project, insert_mng, get_ractangle

# Material list for substrate randomization
MATERIALS = ["\"FR4_epoxy\"", "\"Rogers RO3003 (tm)\"", "\"Rogers RO3006 (tm)\"", "\"Rogers RO3010 (tm)\"", "\"Rogers RO3203 (tm)\"", "\"Rogers RO3210 (tm)\"", "\"Rogers RO4003 (tm)\"", "\"Rogers RO4232 (tm)\"", "\"Rogers RO4350 (tm)\"", "\"Rogers RT/duroid 5870 (tm)\"", "\"Rogers RT/duroid 5880 (tm)\"", "\"Rogers RT/duroid 6002 (tm)\"", "\"Rogers RT/duroid 6006 (tm)\"", "\"Rogers RT/duroid 5880 (tm)\"", "\"Rogers TMM 10 (tm)\"", "\"Rogers TMM 10i (tm)\"", "\"Rogers TMM 3 (tm)\"", "\"Rogers TMM 4 (tm)\"", "\"Rogers TMM 6 (tm)\"", "\"Teflon (tm)\""]

def process_image_data(image_path, data):
    """
    Process image data into binary bitmap
    Returns:
        list: 2D array representing material presence
    """
    img = Image.open(image_path + data)
    img_array = np.array(img)
    
    # Handle different image formats
    if data.startswith("freeform"):
        return [[1 if pixel == 0 else 0 for pixel in row] for row in img_array]
    else:
        return [[1 if pixel[0] == 0 else 0 for pixel in row] for row in img_array]

def modeling_worker(info, thread_id, dataset, args):
    """
    Worker thread for model generation
    Args:
        info: Thread configuration [start_index, end_index]
        thread_id: Worker ID
        dataset: Image dataset to process
        args: Argument parser object
    """
    pythoncom.CoInitialize()
    random.seed(42)  # Ensure reproducible results
    
    hfss_app = win32com.client.Dispatch('AnsoftHfss.HfssScriptInterface')
    desktop = hfss_app.GetAppDesktop()
    
    for idx, data in enumerate(tqdm(dataset[info[thread_id][0]:info[thread_id][1]])):
        # Generate bitmap from image
        bitmap = process_image_data(args.input_path, data)
        
        # Parameter randomization
        scale = round(random.uniform(1, 150), 1)
        copper_height = random.randint(32, max(32, int(16 * scale)))
        material = random.choice(MATERIALS)
        material_height = random.randint(35, max(35, int(16 * scale)))
        
        # Create HFSS project
        project = new_project(
            scale=scale,
            desktop=desktop,
            design_name=data[:-4],
            material=material,
            copper_height=copper_height,
            material_height=material_height
        )
        
        # Insert model geometry
        grouped_rectangles = get_ractangle(bitmap, "simple")
        insert_mng(scale, grouped_rectangles, project, data[:-4], copper_height)
        
        # Save and cleanup
        material_safe = material.replace('\"', '').replace(' ', '_').replace('/', '_')
        size_str = f"({scale},{copper_height},{material_height})"
        save_path = f"{args.output_path}/{data[:-4]}_{size_str}_{material_safe}.aedt"
        
        project.SaveAs(save_path, True)
        desktop.CloseProject(f"{data[:-4]}_{size_str}_{material_safe}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HFSS Automation Tool')
    parser.add_argument('--face', type=int, default=1, 
                       help='Single/Double face configuration')
    parser.add_argument('--input_path', help='path to image dataset', type=str, default="../dataset/")
    parser.add_argument('--output_path', help='path to hfss aedt dataset', type=str, default="../output/")
    parser.add_argument('--thread_num', help='thread number', type=int, default=1)
    args = parser.parse_args()

    dataset = os.listdir(args.input_path)
    done_list = os.listdir(args.output_path)

    thread_config = [[len(done_list), len(dataset)]]  # Thread start/end indices
    threads = []
    
    # Start modeling threads
    for i in range(args.thread_num):
        t = threading.Thread(target=modeling_worker, 
                           args=(thread_config, i, dataset, args))
        t.start()
        threads.append(t)
    
    for t in threads:
        t.join()