import math
import numpy as np
import cv2
import time
import os   
from src import robot
from src import sensor
from src.robot import move2target, move2config
from src.helper import generate_scan_points, \
                        cyl2cart, cart2cyl, \
                        generate_scan_poses, \
                        confirm_defects, plot_surface
from zmqRemoteApi import RemoteAPIClient
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import shutil

class vcell:
    """
    vcell: is the virtual cell in the coppelasim software. It has a robot arm, positioner, proximity sensor, and camera.
    This cell is built to have a dynamic reference frame (positioner as leader and robot arm as follower)
    Notes:
        - keep in mind that sim.handle_world can be referred to as (-1)
    """
    def __init__(self, workpiece_dia, workpiece_len, show_plots_flag) -> None:
        
        # constants
        self.workpiece_dia = workpiece_dia
        self.workpiece_len = workpiece_len
        
        # connect to the coppelasim software, and extract cell info
        self.client = RemoteAPIClient()
        sim = self.client.getObject('sim')
        simIK = self.client.getObject('simIK')
        self.simStep = sim.getSimulationTimeStep()
        
        # When simulation is not running, ZMQ message handling could be a bit
        # slow, since the idle loop runs at 8 Hz by default. So let's make
        # sure that the idle loop runs at full speed for this program:
        defaultIdleFps = sim.getInt32Param(sim.intparam_idle_fps)
        #sim.setInt32Param(sim.intparam_idle_fps, 0)

        # get handle IDs of the cell
        self.wp_id = sim.getObject('/workpiece')
        self.exp_tg_id = sim.getObject('/exp1_tg')
        
        # robot arm class
        self.arm = robot.arm(sim, simIK, 
                             group_name='/robot_grp',
                             base_name='/r1000ia80f',
                             tip_name='/laser_tcp',
                             iktg_name='/IkTarget',
                             home_tg_name='/home_tg')
        
        # positioner class
        self.positioner = robot.positioner(sim, simIK, self.arm.ikEnv, self.arm.ikGroup,
                                           group_name = '/positioner_group', 
                                           tip_name = '/positioner_tcp')
        
        # vision sensor
        self.camera = sensor.camera(sim, sensor_name = '/rgb_camera')
        
        # compute proper standoff distance
        # d/2 + depth view range - tcp diff
        self.camera.dist2tcp = sim.getObjectPose(self.camera.id, self.arm.tip_id)
        self.camera.standoff_dist = (self.workpiece_dia/2) + self.camera.far + self.camera.dist2tcp[2] - 0.001  # 1 mm for the image to be in depth
        self.camera.xshift = - self.camera.dist2tcp[0]
        
        # proximity sensor
        self.tof = sensor.tof(sim, sensor_name= '/tof', 
                              parent_frame = '/tof_frame',
                              measure_frame='/measure_frame')
        
        # store instance of this simulation enviroment
        self.sim = sim
        self.simIK = simIK
        
        # temporary to ensure workpiece is hidden before starting simulation
        #self.sim.setObjectInt32Param(self.wp_id,10,0)
        
        ## cell params
        # compute the minimum number of cycles to scan the part
        self.scan_cycles = math.ceil(self.workpiece_len / self.camera.fovl)
        self.uframe_id = self.sim.getObject('/positioner_tcp')
        #self.defects_handle=self.sim.addDrawingObject(self.sim.drawing_points,10,0,self.uframe_id,100,[1,0,1])
        
        
        # extra params
        self.filename = time.strftime("%Y%m%d-%H%M%S")        
        self.output_dir = os.path.abspath('./output') + '\\'+ self.filename
        self.filepath = self.output_dir + '\\' + self.filename
        os.mkdir(self.output_dir)
        
        # start simulation in the coppeliasim software
        self.startSim()
        print('Cell is ready to run.')

        # show plots flag
        self.show_plots = show_plots_flag
        
    ## -------------------------------- main methods -------------------------------------
    
    # prepare workpiece
    def prepare_part(self) -> None:
        
        # robot go home for the part to load
        self.arm.go_home()
        
        # load workpiece to the scene
        #self.sim.setObjectInt32Param(self.wp_id,10,1)   # (handle, layer, True/False)
        
        # build approach target poses for arm and positioner
        self.scan_posi_cfg , self.scan_arm_trgt = self.build_approach_target(self.camera.half_fovl)
        
        # move positioner to target config
        self.move2sync_cfgpos(target_config=self.scan_posi_cfg, target_pose=[], motion_type="fixed_tcp")
        
        # approach workpiece
        self.arm.approach_part()
        
    # scan workpiece to collect images
    def scan_part(self) -> None:
        print("Scanning workpiece ...")

        # constants
        starting_angle = 0
        finishing_angle = 2*math.pi  #  360 degs
        
        # compute targets for scanning steps
        self.scan_cfgs, self.scan_trgts =  self.compute_scan_poses(init_scan_angl=starting_angle,
                                                                        final_scan_angl=finishing_angle,
                                                                        init_arm_pose= self.scan_arm_trgt)
        
        scan_data = []
        # rotate part and capture pictures per cycle
        for cyc in range(0, self.scan_cycles):
            
            # rotate and scan
            cycle_images, cycle_joint_position_speed = self.scan_cycle(self.scan_cfgs[cyc], self.scan_trgts[cyc])
            
            # collected images, position and speed
            scan_data.append([cycle_images, cycle_joint_position_speed])

        # store data
        self.scan_data = scan_data
        
                
    # map workpiece surface
    def map_surface_forward(self, show_image=False, save_image=True):
        

        cycle_img = []
        angular_position_vectors = []
        mappedSurfacImg = []
        surface_map = []

        for cyc in range(0, self.scan_cycles):
            
            # images per cycle
            imgs_cycle = self.scan_data[cyc][0]                                # list of unit8 images
            angular_position_vectors.append(self.scan_data[cyc][1][0])         # position in degrees
            
            # to obtain the optimal stitching pixels 
            # position curve
            steps = np.linspace(0, len(self.scan_data[cyc][1][1]), len(self.scan_data[cyc][1][1]), True)
            angl_position = (math.pi/2) - np.array(self.scan_data[cyc][1][0])  # 90 - theta
            dtheta = abs(np.diff(angl_position))
            # remove overshooting values
            for i in range(0,len(dtheta)):
                if abs(dtheta[i]) >= math.pi/2:
                    dtheta[i] = dtheta[i-1]
            dtheta = np.append(dtheta, 0)
            m2p = self.camera.px/self.camera.mx             # mapping from m to pixels
            pixp = m2p * (self.workpiece_dia/2 * dtheta)
            plt.plot(steps, pixp)
            plt.xlabel('Frames')
            plt.ylabel('Optimal Stitching Width in (pixels)')
            if self.show_plots:
                plt.show()

            # round to the nearest even number
            avg_pix = math.ceil(np.mean(pixp) / 2.) * 2
            max_pix = avg_pix
            
            # concatination of images per cycle
            midpix = self.camera.px / 2
            idx1 = midpix - 2#(avg_pix/2)
            idx2 = midpix + 2#(avg_pix/2)
            
            # z position 
            dz = (self.camera.my / self.camera.py)
            str_z = self.camera.my * cyc        # start
            stp_z = str_z + self.camera.my      # stop
            z_vector = np.linspace(str_z, stp_z, self.camera.py, endpoint=True)
            
            cumul_pix = 0
            angl_map = []
            getPixels = np.full(len(imgs_cycle), True)
            
            avg_pix = 2     # minimum
            id0 = 0
            cumul_skip = 0
            
            # constant of angle per pixel
            theta_per_pix = 1 / (m2p * (self.workpiece_dia/2))   # mx / (px * r)  ## angle per pix
            
            opt_pix = []
            for i in range(0,len(imgs_cycle)): #img in imgs_cycle:
                
                if getPixels[i]:
                    if (cumul_pix or pixp[i]) >= avg_pix/2:
                        # convert image from unit8 to rgb
                        bgr_img = np.frombuffer(imgs_cycle[i], dtype=np.uint8).reshape(self.camera.py, self.camera.px, 3)
                        rgb_img = cv2.flip(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB), 0)

                        # line image
                        scan_line = rgb_img[:, int(idx1):int(idx2)]
                        
                        if len(cycle_img):
                            cycle_img = cv2.hconcat([scan_line, cycle_img])
                            #angl_map = np.append(angl_map, angl_position[i-int(avg_pix/2):i+int(avg_pix/2)])
                        else:
                            cycle_img = scan_line
                            #angl_map = np.append(angl_map, angl_position[i-int(avg_pix/2):i+int(avg_pix/2)])
                        
                        # angular spatial map 
                        n = scan_line.shape[1]
                        position1 = angl_position[i]-(theta_per_pix*n/2)
                        position2 = angl_position[i]+(theta_per_pix*n/2)
                        angl_map = np.append(angl_map, np.linspace(position1,position2,n,True))
                            
                        # handling indices
                        if cumul_pix >= avg_pix/2:
                            # skip next images equals the avg_pix/2
                            for id in range(id0,len(pixp)):
                                cumul_skip = cumul_skip + abs(pixp[id])
                                if cumul_skip >= avg_pix/2:
                                    break
                            
                            # indices 
                            id1=id0
                            id0=id
                            id2=id
                            
                            # skip the next pixels
                            getPixels[id1:id2] = False
                            
                            cumul_skip = 0
                        
                        cumul_pix = 0
                        
                        # number of pixels to be taken in the next iteration
                        avg_pix = np.clip(math.ceil(pixp[i] / 2.) * 2 , 2, max_pix)
                        idx1 = midpix - (avg_pix/2)
                        idx2 = midpix + (avg_pix/2)
                        opt_pix.append(avg_pix)
                    else:
                        cumul_pix = cumul_pix + pixp[i]
                

            # full surface map
            cr = self.camera.py * cyc
            len_map = len(angl_map)
            for j in range(0, self.camera.py): # stacking virtical pixels
                # [r, theta, z]
                surface_map.append(np.column_stack((np.repeat(self.workpiece_dia/2, len_map),       # r
                                                np.flipud(angl_map),                                           # theta
                                                np.repeat(z_vector[j], len_map))))                  # z

            # scanning cycles concatenation
            if len(mappedSurfacImg):
                mappedSurfacImg = cv2.vconcat([cycle_img, mappedSurfacImg])
            else:
                mappedSurfacImg = cycle_img
            
            # prepare for next cycle
            cycle_img = []
                    
        # Store image
        if show_image:
            cv2.imshow('single view image:', mappedSurfacImg)
            cv2.waitKey(1)

        if save_image:
            cv2.imwrite(self.filepath + '.png', mappedSurfacImg)
            print('saved to:', self.filepath + '.png')
        
        # store output
        self.mappedSurfacImg = mappedSurfacImg 
        self.surface_map = np.flipud(surface_map)  # flip z
        # save image for defect detection
        with open(self.filepath + '_map.npy', "wb+") as f:
            np.save(f, np.flipud(surface_map))

        
    # detect defect
    def detect_defects(self):
        '''
        detects defects on image and returns boxes
        '''
        
        # temp #############
        #self.filename = 'testing'
        #self.output_dir = os.path.abspath('./output') + '\\'+ self.filename
        #self.filepath = self.output_dir + '\\' + self.filename
    
        self.run_path = os.path.abspath('./src/yolov7_model/runs/predict-seg/') + '\\' + self.filename
        relative_img_path = os.path.relpath(self.filepath)
        
        # temp for testing
        with open(self.filepath + '_map.npy', "rb+") as f:
            self.surface_map = np.load(f)
        self.mappedSurfacImg = cv2.imread(self.filepath + '.png')
        ###################
        
        # image
        img = self.mappedSurfacImg
        
        #plot_surface(self.surface_map)
        
        # predict using yolov7
        os.system('python   src\\yolov7_model\\segment\\predict.py \
                            --weights src\yolov7_model\\exported_model\\weights\\yolov7-defect.onnx \
                            --source {path}.png \
                            --view-img \
                            --save-txt \
                            --save-crop \
                            --conf-thres 0.69 \
                            --name {name}'.format(path=relative_img_path, name=self.filename))
        
        # load prediction boxes
        with open(self.run_path + '\\labels\\' + self.filename + '.npy', "rb+") as f:
            predictions = np.load(f)
        
        # remove false prediction (limits measure from top left corrner)
        toplimit = 0
        dif_lim = self.scan_cycles * self.camera.my - self.workpiece_len
        if dif_lim > 0:
            toplimit = dif_lim * self.camera.m2p_y
        lowerlimit = img.shape[0] - math.floor(self.positioner.jaws_length * self.camera.m2p_y) # fixed limit
        confirmed_predictions = confirm_defects(predictions, self.surface_map, inlimits=[toplimit, lowerlimit], area_thr = 0.000025) # 5x5 mm
        
        # store image of final predicted defects
        for box in confirmed_predictions:
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (1,1,1), 5)
        cv2.imwrite(self.filepath + '_confirmed.png', img)
        
        # store bbox
        self.defects_boxes = confirmed_predictions
    
        
    def map_surface_backward(self):
        # mapping the generated shapes from the pervious function
        # to a three-dimentional spitial space
           
        cyl_defects = []
        cart_defects = []     
        for defect in self.defects_boxes:
            # map points to cyl and cart
            # (x1,y1), (x1,y2), (x2, y2), (x2,y1)  # four corrners
            pixs = [defect[0], defect[1], defect[0], defect[3],
                    defect[2], defect[3], defect[2], defect[1]]
            
            # flip indices because coppeliasim and opencv have diff coordiante sys.
            # points are in cartesain coordinate
            points = [self.surface_map[pixs[1],pixs[0]],  # corrner 1
                      self.surface_map[pixs[3],pixs[2]],  # corrner 2
                      self.surface_map[pixs[5],pixs[4]],
                      self.surface_map[pixs[7],pixs[6]]]
                    
            cyl_defects.append(points)
            cart_defects.append(cyl2cart(points))

            self.move2sync_cfgpos([-math.pi/2,0], [], motion_type="fixed_tcp")
            
            # defect corrner visulization 
            for p in cyl2cart(points):
                dummay = self.sim.createDummy(0.01)
                self.sim.setObjectParent(dummay, self.uframe_id)
                ps = self.sim.buildPose(p.tolist(),[0,0,0])
                self.sim.setObjectPose(dummay, self.uframe_id, ps)
                #self.sim.insertPointsIntoPointCloud(self.defect_handle, 0, p.tolist())
                #self.sim.addDrawingObjectItem(self.defects_handle, p.tolist())
                
        # store data
        self.cyl_defects = cyl_defects
        self.cart_defects = cart_defects
        
    # scan individual defects
    def scan_defects(self, scanning_pitch = 0.01):
        '''
        defects: are group of areas that repersents contours around targeted areas.
        '''
        # constants
        self.tof.scanning_pitch = scanning_pitch
        
        # pcd scan path planning
        self.defect_plans = self.plan_3Dscan()
        
        # excutring and collecting pcd data
        self.pcds = self.excute_scan_plan()         # with repsect to user frame (workpiece frame)
        
        # plot
        if self.show_plots:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            pcds = cart2cyl(self.pcds)
            r = pcds[:,0]
            theta = pcds[:,1]
            z = pcds[:,2]
            ax.scatter(r*theta, z, r)
        
        
    def plan_repair():
        pass
    
    
    def repair_part():
        pass
    # --------------------------------- supporting functions -------------------------------------
    # start simulation
    def startSim(self):
        self.sim.startSimulation()
        print('Simulation Started.')
    
      
    # stop simulation
    def stopSim(self):
        self.sim.stopSimulation()
        print('Simulation Stopped.')

    
    def build_approach_target(self, half_fovl):
        '''
        Returns:    positioner goal: is a config goal
                    arm goal: is a target pose goal
        '''
        # positioner config 
        positioner_axis1 = -math.pi/2
        positioner_axis2 = 0
        
        # target pose
        # long_dist = 0.5 field of view (m) + dist of cam to tcp
        long_dist = half_fovl + abs(self.camera.dist2tcp[1])         # distance from poisitioner table to camera center when scanning
        arm_appr_tg_orient = [math.pi/2, 0, 0]                  # with respect to user frame ( dynamic positioner frame ) virtical pose
        
        # built from constants
        positioner_goal = [positioner_axis1, positioner_axis2]       # position in degs of the two joints of positioner
        init_arm_mat = self.sim.buildMatrix([self.camera.xshift, self.camera.standoff_dist, long_dist], arm_appr_tg_orient)   # sim.buildMatrix(position, eularAngles) 
        init_arm_pose = self.sim.matrixToPose(init_arm_mat)
        
        return positioner_goal, init_arm_pose
    
    def compute_scan_poses(self, init_scan_angl, final_scan_angl, init_arm_pose):
        '''
        Inputs: - init_scan_angl: starting scan angle in degs
                - final_scan_angl: last scan angle in degs
                - init_arm_pose: comes from part approach
        '''
        # constants
        p_axis1 = -math.pi/2            # constant (horizontal config)
        p_axis2_0 = init_scan_angl      
        p_axis2_100 = final_scan_angl
        fovl = self.camera.fovl
                
        # create a copy of the positioner tcp for future approach target
        self.approach_id = self.sim.createDummy(0.01)
        self.sim.setObjectParent(self.approach_id, self.positioner.tip_id)     # attach approach to user frame which is positioner tip
        self.sim.setObjectPose(self.approach_id, -1, self.sim.getObjectPose(self.positioner.tip_id, -1)) # -1 handle_world
        
        # move and rotate the approach target outward 2xtimes the max part dia
        self.sim.setObjectPose(self.approach_id, self.approach_id, init_arm_pose)
        
        # initialization
        positioner_scan_goals = []
        arm_scan_goals = []
        for cyc in range(0,self.scan_cycles):

            # positioner poses
            starting_pose = [p_axis1, p_axis2_0]
            final_pose = [p_axis1, p_axis2_100]
            positioner_scan_goals.append([starting_pose, final_pose])            

            # robot arm poses
            offset_distance = fovl * cyc            # pitch distance btw two scans
            trans_mat = self.sim.buildMatrix([0, offset_distance, 0], [0,0,0])      # matrix for transilation of init target
            init_arm_mat = self.sim.poseToMatrix(init_arm_pose)
            new_mat = self.sim.multiplyMatrices(init_arm_mat, trans_mat)
            new_pose = self.sim.matrixToPose(new_mat)
            arm_scan_goals.append(new_pose)
        
        # robot pose is with respect to user frame (positioner tip)
        return positioner_scan_goals, arm_scan_goals
    
    # one scan cycle
    def scan_cycle(self, scan_cfg, scan_trgt):
        '''
        Move the positioner and robot arm to the initial pose, and then trigger capture with motion to final pose.
        Inputs: - self.scan_cfg: the inital and fianl poses of scanning cycle
                - self.scan_trgt: the robot arms poses of the scanning cycle
                - self.approach_id: is the handle for the approach target
        '''

        # move positioner and robot arm to starting pose
        # ensure positioner in correct starting angle
        move2config(self.positioner, scan_cfg[0])
        
        # update approach target pose with scan_trgt, and move robot
        self.sim.setObjectPose(self.approach_id, self.sim.handle_parent, scan_trgt)
        move2target(self.arm, self.sim.getObjectPose(self.approach_id, self.sim.handle_world))
        trg1 =  self.sim.getObjectPose(self.approach_id, self.sim.handle_parent)
        
        
        # excuting capturing images motion, and return images
        collected_images,pcds,aglPosVel,simTime=self.move2sync_cfgpos(target_config = scan_cfg[1], 
                                                            target_pose = [], 
                                                            motion_type = "fixed_tcp",
                                                            refTgtFrame = "handle_world",
                                                            capture_images = True,        # capture images from vision sensor
                                                            collect_pcds = False,          # collect point cloud data using tof
                                                            record_partPosVel = True,     # record workpiece angular position and speed of positioner (scan joint) 
                                                            return_time = False,           # return simulation time
                                                            metric_flag = False)
        
        # images, angular position and velocity
        return collected_images, aglPosVel
    
    
    def plan_3Dscan(self):
        '''
        Input: defect_bboxes are list of boxes contains 4 corrners in [r,theta,z] format
        Output: list of plans points in [r, theta, z] format
        '''

        # generate ordered scanning points (start, end)
        defects_plans = []
        for bbox in self.cyl_defects:
            
            # shift box position in z to match tof sensing point
            for i in range(0,len(bbox)):
                bbox[i][2] = bbox[i][2] + self.tof.tof_dist2tcp
            
            # genere path points, and poses
            plan_points, scan_triggers = generate_scan_points(bbox, self.tof.scanning_pitch)
            plan_poses = generate_scan_poses(plan_points, scan_triggers, self.tof.scanning_res)
            
            defects_plans.append([plan_poses, plan_points, scan_triggers])
        
        # generated ordered scanning points with triggers in [r, theta, z], trigger format
        # points are with respect to the user frame (positioner frame)
        return defects_plans # poses, points, triggers
    

    def excute_scan_plan(self):
        '''
        Inputs:
            - defect_plans: a list of plans contains points and triggers in [r, theta, z] format
        '''

        pcd_scans = []
        approach = True
        for plan in self.defect_plans:
            
            for path in plan[0]:
                cyl_pcd = []
                scan_trigger = path[1]
                
                if approach:
                    # approach part move
                    cfg1 = path[0][0][6:8].tolist()
                    cfg2 = path[0][1][6:8].tolist()
                    pos1 = self.sim.buildPose(path[0][0][0:3].tolist(), path[0][0][3:6].tolist())
                    pos2 = self.sim.buildPose(path[0][1][0:3].tolist(), path[0][1][3:6].tolist())
                    
                    # fix tcp and move 
                    self.move2sync_cfgpos(cfg1,pos1,motion_type="fixed_tcp", refTgtFrame="handle_world", cyclic_positionerJoint=True)
                    # move robot to pos
                    self.move2sync_cfgpos(cfg1,pos1,motion_type="fixed_cfg", refTgtFrame="uframe")
                    
                    approach = False
                    
                if scan_trigger:
                    # move and collect points
                    pcd = self.dynMove(path[0], "uframe", 0, 1, 0, 0, 0)
                    
                    # convert colleted pcd to cyl
                    #cyl_pcd.append(cart2cyl(pcd))
                    cyl_pcd.append(pcd)
                else:
                    # transition move
                    cfg1 = path[0][0][6:8].tolist()
                    cfg2 = path[0][1][6:8].tolist()
                    pos1 = self.sim.buildPose(path[0][0][0:3].tolist(), path[0][0][3:6].tolist())
                    pos2 = self.sim.buildPose(path[0][1][0:3].tolist(), path[0][1][3:6].tolist())
                    
                    # fix tcp and move 
                    self.move2sync_cfgpos(cfg1,pos1,motion_type="fixed_tcp", refTgtFrame="handle_world", cyclic_positionerJoint=True)
                    # move robot to pos
                    self.move2sync_cfgpos(cfg1,pos1,motion_type="fixed_cfg", refTgtFrame="uframe")
                    # fix tcp and move 
                    self.move2sync_cfgpos(cfg2,pos2,motion_type="fixed_tcp", refTgtFrame="handle_world", cyclic_positionerJoint=True)
                    # move robot to pos
                    self.move2sync_cfgpos(cfg2,pos2,motion_type="fixed_cfg", refTgtFrame="uframe")
                    
            
            pcd_scans.append(cyl_pcd)
            
            # approach workpiece using tof
            #move2config(self.positioner, paths[0][0])       # first config point scanning path
            #move2target(self.arm, paths[0][1])              # first pose of paths

        
        # pcd_scans in cyl format with respect to user frame (positioner frame)
        # return in [r,theta,z] format
        return pcd_scans
    
    
    
    ############################# Motion Functions ############################################
    ## This function rquire target pose only and it computes the poses between, path planning is
    ## done using ruckig libaray here. 
    def move2sync_cfgpos(self, target_config, target_pose, 
                         motion_type,
                         refTgtFrame = "handle_world",  # handle_world, handle_parent, uframe
                         capture_images = False,        # capture images from vision sensor
                         collect_pcds = False,          # collect point cloud data using tof
                         record_partPosVel = False,     # record workpiece angular position and speed of positioner (scan joint) 
                         return_time = False,           # return simulation time
                         metric_flag = False,
                         cyclic_positionerJoint = False):  
        '''
        ################## This is non-blocking mode #####################
        A syncretized motion of the positioner and the robot arm in a dynamic configuration reference frame.
        This function works in three different ways: Those options are impelemented in the "motion_type" var.
            1 - "fixed_tcp": (ftp) given positioner configs with fixed tcp >> tcp pose is not given.
            2 - "fixed_cfg": (fcg) given robot pose with a fixed positioner cofig >> positioner is not given. 
            3 - "both": (bth) given positioner configs with moving tcp >> needs both config and tcp targets.
            target_pose: with respect to refTgtFrame, defualt world.
        '''
        # initialization for all options
        images = []
        pcds = []
        aglPosVel = []
        simTime = []
        pJ = []   # joint position vector
        sJ = []   # joint speed vector
        
        # constants
        baseCycleTime = 0.0001
        timeStep = 0
        ruckig_flag = -1
           
        # other params
        self.client.setStepping(True)    # enable stepping mode
        if refTgtFrame == "uframe":
            ref_handle = self.uframe_id
        else:
            ref_handle = getattr(self.sim, refTgtFrame) # reference frame handle
        
        # positioner joints
        pJ1 = self.positioner.joints[0] # axis 1
        pJ2 = self.positioner.joints[1] # axis 2
        
        ## ----------------------------------------------------------
        # 1- maintain tcp pose fixed while moveing positioner to target configuration
        if motion_type == "fixed_tcp" or motion_type == 1:
            
            # temp for plotting
            ps_agl=[]
            robot_agl=[]
            robot_agl_rpw = []
            
            # get current tcp pose
            currTcpPose = self.sim.getObjectPose(self.arm.tip_id, ref_handle)
            currentPos = [self.sim.getJointPosition(pJ1), self.sim.getJointPosition(pJ2)]
            
            # compute config 
            currentPosVelAccel, maxVelAccelJerk, \
            targetPosVel, sel, outPos, outVel, \
                                    outAccel = buildConfigParams(currentPos,
                                                                 [0,0],
                                                                 [0,0],
                                                                 self.positioner.maxV,
                                                                 self.positioner.maxA,
                                                                 self.positioner.maxJrk,
                                                                 target_config,
                                                                 [0,0],
                                                                 [0,int(cyclic_positionerJoint)])
            
            # generate configuration motion using ruckig plug-in
            ruckigObject = self.sim.ruckigPos(self.positioner.nJoints,          #dofs
                                        baseCycleTime,                          #baseCycleTime
                                        ruckig_flag,                            #defualt rucking flag
                                        currentPosVelAccel,                     #currentPosVelAccel
                                        maxVelAccelJerk,                        #maxVelAccelJerk
                                        sel,                                    #selection
                                        targetPosVel)                           #targetPosVel
            
            # initiate response
            result = 0
            timeLeft = 0

            # excute configs motion and mentaiting fixed tcp pose
            # update joints position for each step of the simulation
            while (result == 0):
                dt = timeStep
                if dt == 0:
                    dt = self.sim.getSimulationTimeStep()
                # compute config for the upcoming step
                syncTime = 0 
                result,newPosVelAccel,syncTime = self.sim.ruckigStep(ruckigObject,dt)
                if result >= 0:
                    if result == 0:
                        timeLeft == dt-syncTime 
                    ### move to confing
                    len_joints = self.positioner.nJoints
                    for i in range(0,len_joints):
                        joint_handle = self.positioner.joints[i]
                        outPos[i]=newPosVelAccel[i]
                        outVel[i]=newPosVelAccel[len_joints+i]
                        outAccel[i] =newPosVelAccel[len_joints+i]
                        
                        # apply to the scene
                        if self.sim.isDynamicallyEnabled(joint_handle):
                            self.sim.setJointTargetPosition(joint_handle,outPos[i])
                        else:
                            self.sim.setJointPosition(joint_handle,outPos[i])
                    
                    # update arm pose (set current arm pose)
                    self.sim.setObjectPose(self.arm.iktg_id, ref_handle, currTcpPose)
                    self.arm.simIK.applyIkEnvironmentToScene(self.arm.ikEnv,self.arm.ikGroup)
                    
                    # temp for plotting 
                    ps_agl.append(np.array([self.sim.getJointPosition(pJ1), self.sim.getJointPosition(pJ2)]))
                    robot_agl.append(np.array(self.sim.getObjectOrientation(self.arm.tip_id, self.uframe_id)))
                    al, be, ga = self.sim.getObjectOrientation(self.arm.tip_id, self.uframe_id)
                    w,p,r = self.sim.alphaBetaGammaToYawPitchRoll(al,be,ga)
                    robot_agl_rpw.append(np.array([r,p,w]))
                    
                    '''
                    ps_agl =np.array(ps_agl)
                    robot_agl = np.array(robot_agl)
                    robot_agl_rpw = np.array(robot_agl_rpw)
                    plt.plot(np.linspace(0,len(ps_agl),len(ps_agl),1), ps_agl[:,0], label='J1')
                    plt.plot(np.linspace(0,len(ps_agl),len(ps_agl),1), ps_agl[:,1], label='J2')
                    #plt.plot(np.linspace(0,len(ps_agl),len(ps_agl),1), robot_agl[:,0], '--', label='Roll')
                    #plt.plot(np.linspace(0,len(ps_agl),len(ps_agl),1), robot_agl[:,1], '--', label='Pitch')
                    #plt.plot(np.linspace(0,len(ps_agl),len(ps_agl),1), robot_agl[:,2], '--', label='Yaw')
                    plt.plot(np.linspace(0,len(ps_agl),len(ps_agl),1), robot_agl_rpw[:,0], '--', label='yaw')
                    plt.plot(np.linspace(0,len(ps_agl),len(ps_agl),1), robot_agl_rpw[:,1], '--', label='pitch')
                    plt.plot(np.linspace(0,len(ps_agl),len(ps_agl),1), robot_agl_rpw[:,2], '--', label='roll')
                    plt.legend(loc="upper left")
                    plt.xlabel('Steps')
                    plt.ylabel('Angular position in (rad)')
                    plt.show()
                    '''
                    # capture image from the camera
                    if capture_images:
                        img, pxy = self.sim.getVisionSensorImg(self.camera.id)
                        #self.sim.saveImage(img, [760,1280], 0, "C:/Users/almusaib/Desktop/temp_imgs/test.png", 100)
                        images.append(img)
                    
                    # collect point cloud
                    if collect_pcds:
                        self.sim.read
                        tof_trigger, dist, tof_point, detHandle, normal = self.sim.readProximitySensor(self.tof.id)
                        
                        # change reference frame of detected point to the user frame (positioner leader frame)
                        uf2tofMat = self.sim.getObjectMatrix(self.tof.id, self.uframe_id)
                        tof2ufMat = self.sim.invertMatrix(uf2tofMat)
                        pointUF = self.sim.multiplyVector(tof2ufMat, tof_point)
                        pcds.append(pointUF)   # x, y, z with user frame
                    
                    # record angular position and speed of the workpiece    
                    if record_partPosVel:
                        partJoint_handle = self.positioner.joints[1]     # axis2
                        pJ.append(self.sim.getJointPosition(partJoint_handle))
                        sJ.append(self.sim.getJointVelocity(partJoint_handle))
                    
                    # return simulation time at the current step
                    if return_time:
                        simTime.append(self.getSimulationTime())
                else:
                    raise RuntimeError("sim.ruckigStep returned error code "+result)
    
                if result==0:    
                    # triggers next simulation step
                    self.client.step()       
            
            # distroy ruckig object when finish
            self.sim.ruckigRemove(ruckigObject)
        
        # 2 - move tcp to pose with positioner config fixed.
        if motion_type == "fixed_cfg" or motion_type == 2:
            
            # constants
            sel = [1,1,1,1]   # 4 DOFs acepts any values other than 0 
            
            # some of the code below is adopted from the coppeliasim zmqapi
            # this motion is consider a motion around an arbitrary axis
            
            # get current positioner config and tcp pose
            currCfg = [self.sim.getJointPosition(pJ1), self.sim.getJointPosition(pJ2)]
            currTcpPose = self.sim.getObjectPose(self.arm.tip_id, ref_handle)
            
            maxVel = self.arm.maxV
            maxAccel = self.arm.maxA
            maxJerk = self.arm.maxJrk
            
            usingMatrices = (len(currTcpPose)>=12)
            if usingMatrices:
                currentMatrix = currTcpPose
                targetMatrix = target_pose
            else:
                currentMatrix = self.sim.buildMatrixQ(currTcpPose,[currTcpPose[3],currTcpPose[4],currTcpPose[5],currTcpPose[6]])
                targetMatrix = self.sim.buildMatrixQ(target_pose,[target_pose[3],target_pose[4],target_pose[5],target_pose[6]])

            # get axis and angle 
            outMatrix = self.sim.copyTable(currentMatrix)
            axis, angle = self.sim.getRotationAxis(currentMatrix,targetMatrix)
            timeLeft = 0
            
            # to move in a straight line between two poses
            if metric_flag:
                # Here we treat the movement as a 1 DoF movement, where we simply interpolate via t between
                # the start and goal pose. This always results in straight line movement paths
                
                # constants
                metric = [0.1,0.1,0.1,0.1]
                currentPosVelAccel = [0,0,0]
                
                dx = [(targetMatrix[3]-currentMatrix[3])*metric[0],(targetMatrix[7]-currentMatrix[7])*metric[1],(targetMatrix[11]-currentMatrix[11])*metric[2],angle*metric[3]]
                distance = math.sqrt(dx[0]*dx[0]+dx[1]*dx[1]+dx[2]*dx[2]+dx[3]*dx[3])
                
                if distance > 0.000001:
                    maxVelAccelJerk = [maxVel[0],maxAccel[0],maxJerk[0]]
                    if len(maxVel) > 1:
                        maxVelAccelJerk.append(maxVel[1])
                    if len(maxAccel) > 1:
                        maxVelAccelJerk.append(maxAccel[1])
                    targetPosVel = [distance,0]
                    ruckigObject = self.sim.ruckigPos(1,0.0001,ruckig_flag,currentPosVelAccel,maxVelAccelJerk,[1],targetPosVel)
                    result = 0
                    while (result == 0):
                        dt = timeStep
                        if dt == 0:
                            dt = self.sim.getSimulationTimeStep()
                        result,newPosVelAccel,syncTime = self.sim.ruckigStep(ruckigObject,dt)
                        if result >= 0:
                            if result == 0:
                                timeLeft = dt-syncTime
                            t = newPosVelAccel[0]/distance
                            outMatrix = self.sim.interpolateMatrices(currentMatrix,targetMatrix,t)
                            nv = [newPosVelAccel[1]]
                            na = [newPosVelAccel[2]]
                            if not usingMatrices:
                                q = self.sim.getQuaternionFromMatrix(outMatrix)
                                outMatrix = [outMatrix[3],outMatrix[7],outMatrix[11],q[0],q[1],q[2],q[3]]

                            # apply motion to the scene and return data
                            self.sim.setObjectPose(self.arm.iktg_id, ref_handle, outMatrix)
                            self.arm.simIK.applyIkEnvironmentToScene(self.arm.ikEnv,self.arm.ikGroup)
            
                        else:
                            raise RuntimeError("sim.ruckigStep returned error code "+ result)
                        
                        if result==0:    
                            # triggers next simulation step
                            self.client.step()       
                    
                    self.sim.ruckigRemove(ruckigObject)
            else:
                # Here we treat the movement as a 4 DoF movement, where each of X, Y, Z and rotation
                # is handled and controlled individually. This can result in non-straight line movement paths,
                # due to how the Ruckig functions operate depending on 'flags'
                currentPosVelAccel = [0,0,0,0,0,0,0,0,0,0,0,0]
                dx = [targetMatrix[3]-currentMatrix[3],targetMatrix[7]-currentMatrix[7],targetMatrix[11]-currentMatrix[11],angle]
                maxVelAccelJerk = [maxVel[0],maxVel[1],maxVel[2],maxVel[3],maxAccel[0],maxAccel[1],maxAccel[2],maxAccel[3],maxJerk[0],maxJerk[1],maxJerk[2],maxJerk[3]]
                if len(maxVel) > 4:
                    for i in range(len(maxVel)-len(maxJerk)):
                        maxVelAccelJerk.append(maxVel[len(maxJerk)+i])
                if len(maxAccel) > 4:
                    for i in range(len(maxAccel)-len(maxJerk)):
                        maxVelAccelJerk.append(maxAccel[len(maxJerk)+i])
                targetPosVel = [dx[0],dx[1],dx[2],dx[3],0,0,0,0,0]
                ruckigObject = self.sim.ruckigPos(4,0.0001,ruckig_flag,currentPosVelAccel,maxVelAccelJerk,[1,1,1,1],targetPosVel)
                result = 0
                while (result == 0):
                    dt = timeStep
                    if dt == 0:
                        dt = self.sim.getSimulationTimeStep()
                    result,newPosVelAccel,syncTime = self.sim.ruckigStep(ruckigObject,dt)
                    if result >= 0:
                        if result == 0:
                            timeLeft = dt-syncTime
                        t = 0
                        if abs(angle)>math.pi*0.00001:
                            t = newPosVelAccel[3]/angle
                        outMatrix = self.sim.interpolateMatrices(currentMatrix,targetMatrix,t)
                        outMatrix[3] = currentMatrix[3]+newPosVelAccel[0]
                        outMatrix[7] = currentMatrix[7]+newPosVelAccel[1]
                        outMatrix[11] = currentMatrix[11]+newPosVelAccel[2]
                        nv = [newPosVelAccel[4],newPosVelAccel[5],newPosVelAccel[6],newPosVelAccel[7]]
                        na = [newPosVelAccel[8],newPosVelAccel[9],newPosVelAccel[10],newPosVelAccel[11]]
                        if not usingMatrices:
                            q = self.sim.getQuaternionFromMatrix(outMatrix)
                            outMatrix = [outMatrix[3],outMatrix[7],outMatrix[11],q[0],q[1],q[2],q[3]]

                        # apply motion to the scene and return data
                        self.sim.setObjectPose(self.arm.iktg_id, ref_handle, outMatrix)
                        self.arm.simIK.applyIkEnvironmentToScene(self.arm.ikEnv,self.arm.ikGroup)
                        
                    else:
                        raise RuntimeError("sim.ruckigStep returned error code "+result)
                    
                    if result==0:    
                        # triggers next simulation step
                        self.client.step()
                           
                self.sim.ruckigRemove(ruckigObject)
             
        # 3 - move configuration and robot tcp at the same simulation step    
        if motion_type == "both" or motion_type == 3:
            pass 
            
        # return defualt params
        self.client.setStepping(False)
        
        aglPosVel = [pJ,sJ]
        # return some data from actions
        return images, pcds, aglPosVel, simTime


    ### --------------------------------------- DYNMOVE ----------------------------
    ### for poses that are pre-computed for both robot and positioner (non-blocking)
    def dynMove(self,   targetPoses, 
                        refTgtFrame = "handle_world",  # handle_world, handle_parent, uframe
                        capture_images = False,        # capture images from vision sensor
                        collect_pcds = False,          # collect point cloud data using tof
                        record_partPosVel = False,     # record workpiece angular position and speed of positioner (scan joint) 
                        return_time = False,           # return simulation time
                        cyclic_positionerJoint = False):  

        # initialization for all options
        images = []
        pcds = []
        aglPosVel = []
        simTime = []
        
        # constants
        baseCycleTime = 0.0001
        timeStep = 0
        ruckig_flag = -1
            
        # other params
        self.client.setStepping(True)    # enable stepping mode
        if refTgtFrame == "uframe":
            ref_handle = self.uframe_id
        else:
            ref_handle = getattr(self.sim, refTgtFrame) # reference frame handle
        
        # positioner joints
        pJ1 = self.positioner.joints[0] # axis 1
        pJ2 = self.positioner.joints[1] # axis 2
        
        # init 
        pJ = []   # joint position vector
        sJ = []   # joint speed vector
            
        # get current tcp pose
        #currTcpPose = self.sim.getObjectPose(self.arm.tip_id, self.sim.handle_world)
        #currentPos = [self.sim.getJointPosition(pJ1), self.sim.getJointPosition(pJ2)]

        # excute configs motion and mentaiting fixed tcp pose
        # update joints position for each step of the simulation
        for pose in targetPoses:
            tcpPose = self.sim.buildPose(pose[0:3].tolist(),pose[3:6].tolist())  # building pose
            positionerPose = pose[6:8].tolist()
            
            # move to confing in positioner case
            len_joints = self.positioner.nJoints
            for i in range(0,len_joints):
                joint_handle = self.positioner.joints[i]
                # apply to the scene
                if self.sim.isDynamicallyEnabled(joint_handle):
                    self.sim.setJointTargetPosition(joint_handle,positionerPose[i])
                else:
                    self.sim.setJointPosition(joint_handle,positionerPose[i])
            
            # move arm to the target pose
            self.sim.setObjectPose(self.arm.iktg_id, ref_handle, tcpPose)
            self.arm.simIK.applyIkEnvironmentToScene(self.arm.ikEnv,self.arm.ikGroup)
            
            # capture image from the camera
            if capture_images:
                img, pxy = self.sim.getVisionSensorImg(self.camera.id)
                images.append(img)
            
            # collect point cloud
            if collect_pcds:
                tof_trigger, dist, tof_point, detHandle, normal = self.sim.readProximitySensor(self.tof.id)
                
                # change reference frame of detected point to the user frame (positioner leader frame)
                uf2tofMat = self.sim.getObjectMatrix(self.tof.id, self.uframe_id)
                self.sim.invertMatrix(uf2tofMat)
                pointUF = self.sim.multiplyVector(uf2tofMat, tof_point)
                pcds.append(pointUF)   # x, y, z with user frame
            
            # record angular position and speed of the workpiece    
            if record_partPosVel:
                partJoint_handle = self.positioner.joints[1]     # axis2
                pJ.append(self.sim.getJointPosition(partJoint_handle))
                sJ.append(self.sim.getJointVelocity(partJoint_handle))
            
            # return simulation time at the current step
            if return_time:
                simTime.append(self.getSimulationTime())

            # triggers next simulation step
            self.client.step()       
            
        # return data per request
        if capture_images:
            return images
        if collect_pcds:
            return pcds
        if record_partPosVel:
            return aglPosVel
        if return_time:
            return simTime
            
           
def buildConfigParams(currentPos, 
                      currentVel,
                      currentAccel,
                      maxVel,
                      maxAccel,
                      maxJerk,
                      targetPos,
                      targetVel,
                      cyclicJoints=None):

    # initi
    currentPosVelAccel=[]
    maxVelAccelJerk=[]
    targetPosVel=[]
    sel=[]
    outPos=[]
    outVel=[]
    outAccel=[]
        
    # build pose and velocity
    for i in range(len(currentPos)):
        v=currentPos[i]
        currentPosVelAccel.append(v)
        outPos.append(v)
        maxVelAccelJerk.append(maxVel[i])
        w=targetPos[i]
        if cyclicJoints and cyclicJoints[i]:
            while w-v>=math.pi*2:
                w=w-math.pi*2
            while w-v<0:
                w=w+math.pi*2
            if w-v>math.pi:
                w=w-math.pi*2
        targetPosVel.append(w)
        sel.append(1)
    
    # build velocity and acceleration
    for i in range(len(currentPos)):
        if currentVel:
            currentPosVelAccel.append(currentVel[i])
            outVel.append(currentVel[i])
        else:
            currentPosVelAccel.append(0)
            outVel.append(0)
        maxVelAccelJerk.append(maxAccel[i])
        if targetVel:
            targetPosVel.append(targetVel[i])
        else:
            targetPosVel.append(0)
    for i in range(len(currentPos)):
        if currentAccel:
            currentPosVelAccel.append(currentAccel[i])
            outAccel.append(currentAccel[i])
        else:
            currentPosVelAccel.append(0)
            outAccel.append(0)
        maxVelAccelJerk.append(maxJerk[i])

    if len(maxVel) > len(currentPos):
        for i in range(len(maxVel)-len(currentPos)):
            currentPosVelAccel.append(maxVel[len(currentPos)+i])
    if len(maxAccel) > len(currentPos):
        for i in range(len(maxAccel)-len(currentPos)):
            currentPosVelAccel.append(maxAccel[len(currentPos)+i])
         
    return currentPosVelAccel, maxVelAccelJerk, targetPosVel, sel, outPos, outVel, outAccel
        