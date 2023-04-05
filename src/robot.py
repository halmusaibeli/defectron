import math
import numpy as np
import time as t

## -------------------------------- robot arm -----------------------------------
class arm:
    
    def __init__(self, sim, simIK, group_name, base_name, tip_name, iktg_name, home_tg_name) -> None:
        
        # constants
        self.nJoints = 6   # 6DOFs
        self.maxV = [4.5,4.5,4.5,4.5,4.5,4.5]  # vx,vy,vz in m/s, Vtheta is rad/s
        self.maxA = [1.24,1.24,1.24,1.24,1.24,1.24] # ax,ay,az in m/s^2, Atheta is rad/s^2
        self.maxJrk = [0.1,0.1,0.1,0.1,0.1,0.1] # is ignored (i.e. infinite) with RML type 2
        
        # intiate IDs
        self.grp_id = sim.getObject(group_name)                  # could be frame
        self.base_id = sim.getObject(group_name + base_name)
        self.tip_id = sim.getObject(group_name + tip_name)       # tcp
        self.iktg_id = sim.getObject('/positioner_group' + iktg_name)     # target will be used for ik calculations
        self.home_tg = sim.getObject(home_tg_name)
        self.app_tg = sim.getObject('/app_tg')
        
        # joints ids
        pJoints = []
        for i in range(0,self.nJoints):
            j_name = '{name}/joint_{i}'.format(name=group_name, i = (i+1))
            pJoints.append(sim.getObject(j_name))
        self.joints = pJoints

        
        # Arm IK environment
        ikEnv = simIK.createEnvironment()
        ikGroup = simIK.createIkGroup(ikEnv)
        #simIK.setIkGroupCalculation(ikEnv,ikGroup,simIK.method_pseudo_inverse,1,99)
        ikElement, simToIkObjectMap = simIK.addIkElementFromScene(ikEnv, ikGroup, self.base_id, self.tip_id, self.iktg_id, simIK.constraint_pose) 
        simIK.setIkElementPrecision(ikEnv,ikGroup,ikElement,(0.0005,(0.005*math.pi/180)))
        
        # store vars
        self.sim = sim
        self.simIK = simIK
        self.ikEnv = ikEnv
        self.ikGroup = ikGroup
        self.ikElement = ikElement
    
    # moving robot to home
    def go_home(self):
        
        # get target pose and go home
        homePose = self.sim.getObjectPose(self.home_tg,-1)
        move2target(self, homePose)  # blocking mode
    
    # moving robot to approach workpiece
    def approach_part(self):
        
        # get target and excute motion
        appPose = self.sim.getObjectPose(self.app_tg, -1)
        move2target(self, appPose)
        
## -------------------------------- positioner ------------------------------------
class positioner:
    
    
    def __init__(self, sim, simIk, ikEnv, ikGroup, group_name, tip_name) -> None:
        
        # constants
        self.nJoints = 2   # 2DOFs
        self.maxV = [0.45,0.45,0.45,4.5]  # vx,vy,vz in m/s, Vtheta is rad/s
        self.maxA = [0.13,0.13,0.13,1.24] # ax,ay,az in m/s^2, Atheta is rad/s^2
        self.maxJrk = [0.1,0.1,0.1,0.2] # is ignored (i.e. infinite) with RML type 2
        
        # intiate IDs
        self.sim = sim
        self.simIK = simIk
        self.ikEnv = ikEnv
        self.ikGroup = ikGroup
        self.id = sim.getObject(group_name)
        self.tip_id = sim.getObject(tip_name)
        
        # joints ids
        pJoints = []
        for i in range(0,self.nJoints):
            j_name = '{name}/joint_{i}'.format(name=group_name, i = (i+1))
            pJoints.append(sim.getObject(j_name))
        self.joints = pJoints
        
        # extra 
        self.jaws_length = 0.075  # mm
        
        # joint recording script id
        self.joint2_script_id = self.sim.getScript(1, self.joints[1])  # (child script, obj_id)

## --------------------- common functions -----------------------------------

# move mechanesim using FK (blocking mode)
def move2config(self, configGoal):
    '''
        - This function is a blocking function were the API will wait until
        the movement is computed and executed.
        - This function will update both the leader and follower (positioner and arm).
    '''
    # excuted function at every step
    def moveToConfig_callback(config, v, a, joints): # (currentConfig, velocity, acceleration, auxData)
        
        # update joints position for each step of the simulation
        for i in range(0,self.nJoints):
            j = joints[i]
            if self.sim.isDynamicallyEnabled(j):
                self.sim.setJointTargetPosition(j,config[i])
            else:
                self.sim.setJointPosition(j,config[i])
        
        # update ikEnv of the arm
        self.simIK.applyIkEnvironmentToScene(self.ikEnv,self.ikGroup)
            
    # current joints config
    currnetJointsConfig = []
    for i in range(0,self.nJoints):
        currnetJointsConfig.append(self.sim.getJointPosition(self.joints[i]))
    
    # compute and execute movements from current config to the target
    self.sim.moveToConfig(-1, currnetJointsConfig, None, None, self.maxV, self.maxA, self.maxJrk,
                    configGoal, None, moveToConfig_callback, self.joints, None)


# move to target via IK (blocking mode)
# this is used when motion from one pose to another is required without any other actions
def move2target(self, targetPose): # target: pose or matrix of the target with respect to world

    # excuted function at every step
    def moveToPose_callback(targetPose, v, a, iktg_id): # (currentConfig, velocity, acceleration, auxData)
        self.sim.setObjectPose(iktg_id, self.sim.handle_world, targetPose)
        self.simIK.applyIkEnvironmentToScene(self.ikEnv,self.ikGroup)
                    
    # current tip pose
    currentTipPose = self.sim.getObjectPose(self.tip_id, self.sim.handle_world)
    
    # move to target using IK
    self.sim.moveToPose(-1, currentTipPose, self.maxV, self.maxA, self.maxJrk,
                        targetPose, moveToPose_callback, self.iktg_id, None)
    
