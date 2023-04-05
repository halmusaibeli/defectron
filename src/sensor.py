import numpy as np
import cv2

# camera
class camera():
    
    def __init__(self, sim, sensor_name, fova = None, fovl = None) -> None:
        
        # constants
        fovl = 0.2105   # m (temp)
            
        
        # initiate IDs
        self.id = sim.getObject(sensor_name)
        self.cap_script_id = sim.getScript(sim.scripttype_childscript, self.id) 

        # sensor parameters
        if fova == None:
            self.pta = sim.getObjectFloatParam(self.id, sim.visionfloatparam_perspective_angle)     # in degs
        else:
            self.fova = fova
        
        if fovl == None:
            self.fovl = fovl   # change it with equation
        else:
            self.fovl = fovl
        # distance from camera caneter to table (half the longitudinal camera view)
        self.half_fovl = self.fovl / 2
        
        
        # get pixels
        self.px = sim.getObjectInt32Param(self.id, sim.visionintparam_resolution_x)
        self.py = sim.getObjectInt32Param(self.id, sim.visionintparam_resolution_y)
        

        # compute standoff distance
        self.near = sim.getObjectFloatParam(self.id, sim.visionfloatparam_near_clipping)
        self.far = sim.getObjectFloatParam(self.id, sim.visionfloatparam_far_clipping)
        self.dv = self.far - self.near   # depth view range
        
        # parameters will be computed later
        self.standoff_dist = []    # standoff distance to workpiece when scanning
        self.dist2tcp = []         # distance from camera frame to tcp
        self.xshift = []
        
        # build camera matrix for localization
        self.mx = 0.125   # 760 pixels to m
        self.my = 4/19   # 1250 pixels to m
        
        # ideal pixel mapping
        self.m2p_x = self.px / self.mx
        self.m2p_y = self.py / self.my
        self.p2m_x = 1/self.m2p_x
        self.p2m_y = 1/self.m2p_y
        
        #self.intrx = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        self.extrx = np.array([])
        #self.cam_mat = 
    def backward_mapping(self):
        pass
        #
        
    def hand_eye_calibertation():
        # finding transformation from camera to tcp
        
        transf_cam2tcp = cv2.calibrateHandEye()
        return transf_cam2tcp
# tof
class tof():
    
    def __init__(self, sim, sensor_name, parent_frame, measure_frame) -> None:
        
        
        
        # initialization
        self.id = sim.getObject(sensor_name)
        self.frame_id = sim.getObject(parent_frame)
        self.tof_measure = sim.getObject(measure_frame)
        
        
        # tof params
        self.tof_dist2tcp = 0.1     # m
        self.scanning_res= 0.005    # in rad
        self.scanning_pitch = 0     # default distance between tow traces of scans