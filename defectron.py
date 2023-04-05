'''
The main entry point is a script representing automatic defect search on metal cylindrical components.
    - generate image scanning paths, and execute them.
    - generate single view image of the part.
    - detect defects on the single image and return polygon of the segmentation.
    - ..TODO:: fit primitive shape to the returned polygon.
    - map (localize) point primitve shapes in spatial coordinate system.
    - generate tof scanning paths, execute them.
    - ..TODO:: generate repair paths in Matlab
    
    Notes:
    - unit are in SI, except angle is in degrees
'''
import sys
from tqdm import tqdm
from src import cell


    
# user input:
workpiece_dia = 0.095       # extreme diameter in meters
workpiece_len = 0.450       # len in meters
scanning_pitch = 0.005      # 5 mm, distance in z btw two scanning traces
show_plots_flag = False     # show optimal stitching plots

with tqdm(total=100, file=sys.stdout) as pbar:

    # constants:


    # initiate virtual cell:
    vc = cell.vcell(workpiece_dia, workpiece_len, show_plots_flag)
    
    # prepare cell and load workpiece
    pbar.set_description("Getting part ready for inspection")
    vc.prepare_part()
    pbar.update(5)
    
    # image scan paths generation
    pbar.set_description("Scanning part")
    vc.scan_part()
    pbar.update(30)
    
    # single image generation
    pbar.set_description("Mapping scanned surface")
    vc.map_surface_forward()
    pbar.update(45)
    
    # recognize defects
    pbar.set_description("Detecting defects")
    vc.detect_defects()
    pbar.update(55)
    
    # mapping segementation shapes to spitial space
    pbar.set_description("Localizing defects")
    vc.map_surface_backward()
    pbar.update(60)
    
    # pcd scan paths
    pbar.set_description("Scanning defects")
    vc.scan_defects(scanning_pitch)
    pbar.update(70)
    
    # repair paths
    pbar.set_description("Generating repair paths")
    vc.plan_repair()
    pbar.update(80)
    
    # repair simulation
    pbar.set_description("Repairing part")
    vc.repair_part()
    pbar.update(100)
    
    print('Repair Completed.')
    
    vc.pcds