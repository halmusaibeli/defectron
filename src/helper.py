import math
import numpy as np
import cv2
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import open3d as o3d

def slice_image(image, output_size):
    ########################## Deprecated function ##################################
    
    # ------------> y  (cols)
    # |
    # |
    # |
    # v
    # x 
    # (rows)
    
    # parameters
    px = output_size[0]
    py = output_size[1]
    img_px = image.shape[0]   # rows
    img_py = image.shape[1]   # cols
    
    # compuate number of slices in x direction (rows)
    inter_pxls_x  = img_px - 2 * px                             # internal pixels in x
    inter_m = math.ceil(inter_pxls_x / px)                      # internal pixel
    offset_x = int(px - ( (inter_m * px) - inter_pxls_x ) / 2)            # computed internal length
    total_m = inter_m + 2                                       # interal slices + outer slices
    
    # compute number of slices in y direction (cols)
    inter_pxls_y  = img_py - 2 * py                             # internal pixels in x
    inter_n = math.ceil(inter_pxls_y / py)                      # internal pixel
    offset_y = int(py - ( (inter_n * py) - inter_pxls_y ) / 2)            # computed internal length
    total_n = inter_n + 2                                       # interal slices + outer slices
    
    # initiate first and last slices
    rows, cols = (total_m, total_n)
    slices = {}
    inds = {}
    
    # first and last slices in top row
    slices[0,0] = image[0:px, 0:py]                     # top left image
    slices[0,img_py] = image[0:px, img_py-py-1:-1]      # top right
    slices[img_px,0] = image[img_px-px-1:-1, 0:py]      # bottom left
    slices[img_px,img_py] = image[img_px-px-1:-1, img_py-py-1:-1]  # bottom right
    #cv2.imshow('slc', slices[0,0])
    
    inds[0,0] = [0,px,0,py]
    inds[0,img_py] = [0,px,img_py-py-1,-1]
    inds[img_px,0] = [img_px-px-1, -1, 0, py]
    inds[img_px,img_py] = [img_px-px-1, -1, img_py-py-1,-1]
    
    # slice image
    for row in range(0,inter_m+1):
        for col in range(0,inter_n+1):
            if row == col:
                continue
            else:
                xwin = offset_x+(px*row)
                ywin = offset_y+(py*col)
                slices[row,col] = image[xwin:xwin+px, ywin:ywin+py]
            inds[row, col] = [xwin, px, ywin, py]
            cv2.imwrite('./output/slices/{r}_{c}.png'.format(r= row, c =col),slices[row,col])        
    print('Image has be sliced.')
    return slices, inds

def combine_masks(slices, slicing_inds):
    ######################################## Deprecated function #######################################
    full_img = {}
    rows, cols = slices.shape()
    for row in range(0,rows):
        for col in range(0,cols):
            indx = slicing_inds[row,col]
            for i in range(0,indx[1]):
                for j in range(0,indx[3]):
                    if full_img[i][j] < slices[row,col][i][j]:
                        full_img[i][j] = slices[row,col][i][j]
                    else:
                        full_img[i][j] = 0
                        
                        # because they overlap, the next line will overwrite any exisiting segementation
                        #full_img[indx[0]:indx[1], indx[2]:indx[3]] = slices[row,col]
        
    print('Slices have been fused.')
    return full_img


def generate_scan_points(corrners, pitch, plot=False):
    
    '''
    Algorithm is as following :
        - generate mid-line of the two extremes
        - generate points in line
        - project points on the bbox along longitudinal direction
        - start plan from the minimum point in long. direction (z)
        - return point in wrapped format of cyl with triggers
    '''
    
    def samePointLoc(value, oneD_array, rtol=1e-07, atol=1e-08):
        '''
        Looking for index of same value in a list of values.
        '''
        idx = np.where(np.isclose(value, oneD_array, rtol, atol))[0]
        if len(idx) != 0:
            idx = idx[0]
        else:
            idx = None
        return idx

    #def find_in_array(x, array):
    #    for value in array:
    
    # convert to arc length (unwrapped surface) (r, r*theta, z)
    corrners = np.array(corrners)
    avg_r = np.mean(corrners[:,0], axis=0)
    corrners[:,1] = avg_r * corrners[:,1]   # theta cols
    
    # generate mid-line with longitudinal axis (z axis which indx 2)
    z = corrners[:,2]
    closeIndx = np.argmin(z)
    farIndx = np.argmax(z)
    mline = [z[closeIndx], z[farIndx]]  # mid-line in longitudinal direction of the workpiece
    
    # interpolate line, and create points in averaged theta
    leng = mline[1] - mline[0]
    n = math.ceil(leng / pitch)
    z_inter_points = np.linspace(mline[0], mline[1], n, endpoint=True)
    avg_theta = np.mean(corrners[:,1], axis=0)
    inter_points = np.vstack((z_inter_points, np.repeat(avg_theta, len(z_inter_points)))).T

    # project points on edges of bbox (4 edges)
    projPs = np.empty((0,2))
    for i in range(0,4):
        # ignores r for now and reorger it [z, r*theta]
        box_edge = [np.flipud(corrners[i-1][1:3]), np.flipud(corrners[i][1:3])]
        # intermediate points [z, r*theta]
        proj_points = project_points_onto_lineSeg(inter_points, box_edge) 
        if proj_points.size != 0:
            projPs = np.append(projPs, proj_points, axis=0)
        
    # pick the most left point of the bbox on any edge
    idxr,idxc = np.argmin(projPs, axis=0)  # looking in the first values which are (z) values
    temp_points = projPs
    
    # start planning from the first selected point above
    slc_point = temp_points[idxr]
    path_points = np.array([slc_point])
    temp_points = np.delete(temp_points, idxr, axis=0)  # delete point from list
    trgs = []
    
    # planning alogrithm
    while temp_points.size:
        # find the next point of scanning, it does have same value of axis=0 as selected point
        # looking for equal value at axis=0 which is at same scanning pitch.
        # looking for same x value in list of x values
        idx = samePointLoc(slc_point[0], temp_points[:,0])
           
        # if collinear point exist then add it to scanning path
        if idx != None:
            slc_point = temp_points[idx]
            temp_points = np.delete(temp_points, idx, axis=0)
            trg = True   # this to trigger collection of pcd points moving to the next point
        # if not then look for transition nearest transition point
        else:
            # search for nearest neightbor
            tree = KDTree(temp_points)          # tree of the remaining points
            idx = tree.query(slc_point)[1]
            slc_point = temp_points[idx]        # return the nearest point
            temp_points = np.delete(temp_points, idx, axis=0)
            trg = False  # don't collect pcd points because path is not in defective region (transition)
        
        path_points = np.vstack((path_points, slc_point))
        trgs.append(trg)
    
    # attach r back, and map surface back (wrapped surface)
    ordered_scan_points = []
    for p in path_points:
        # in: [z, r*theta], out: [r, theta, z]
        ordered_scan_points.append([avg_r, p[1]/ avg_r, p[0]])
    
    if plot:
        # plot bounding box
        xs = np.append(corrners[:,1] / avg_r, corrners[0,1] / avg_r)
        ys = np.append(corrners[:,2] * 1000 , corrners[0,2] * 1000 )
        plt.plot(xs,ys, Color='0.8', alpha=0.5)
        
        for i in range(0,len(path_points)-1):
            style='o--'
            color='k'
            if trgs[i]:
                style='o-'
                color='#1f77b4'
            plt.plot(path_points[i:i+2,1], path_points[i:i+2,0],style,Color=color)
        plt.xlabel('angle in rad')
        plt.ylabel('z in mm')
        plt.show()
    
    return [np.array(ordered_scan_points), np.array(trgs, dtype=bool)]
    
        

def project_points_onto_lineSeg(points, lineSeg):
    # this function project points onto edge along vector, which is y axis
    # lineSeg means line segements
    
    # intersection between line(p1, p2) and line(p3, p4)
    def intersect(p1, p2, p3, p4):
        # adopted from https://gist.github.com/kylemcdonald/6132fc1c29fd3767691442ba4bc84018
        x1,y1 = p1
        x2,y2 = p2
        x3,y3 = p3
        x4,y4 = p4
        denom = (y4-y3)*(x2-x1) - (x4-x3)*(y2-y1)
        if denom == 0: # parallel
            return None
        ua = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / denom
        if ua < 0 or ua > 1: # out of range
            return None
        ub = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / denom
        if ub < 0 or ub > 1: # out of range
            return None
        x = x1 + ua * (x2-x1)
        y = y1 + ua * (y2-y1)
        return (x,y)
    
    '''
    def lineSegs_intersection(seg1, seg2):
        # convert to tuple
        xdiff = (seg1[0][0] - seg1[1][0], seg2[0][0] - seg2[1][0])
        ydiff = (seg1[0][1] - seg1[1][1], seg2[0][1] - seg2[1][1])

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        # compute determinant
        div = det(xdiff, ydiff)
        
        if div == 0:
            # no intersection
            return None
        else:
            d = (det(*seg1), det(*seg2))
            x = det(d, xdiff) / div
            y = det(d, ydiff) / div
            return [x, y]
    '''
    
    # edge length using Euclidean distance
    edp1 = np.array(lineSeg[0])
    edp2 = np.array(lineSeg[1])
    sum_sq = np.sum(np.square(edp1 - edp2))    # finding sum of squares
    edg_leng = np.sqrt(sum_sq)

    # computing square root
    projected_points = []
    for p in points:
        # generate segment line from the target project point (10 x bbox long)
        step = 10 * edg_leng
        stp = [p[0], p[1] + step]    # start point
        edp= [p[0], p[1] - step]     # end point
        
        # projection segment
        proj_seg = [stp,edp]
        
        # find intersection between two line segments
        inters = intersect(proj_seg[0], proj_seg[1], lineSeg[0], lineSeg[1])
        if inters:
            projected_points.append(inters)  # store point of intersection if any
                  
    return np.array(projected_points)


def cyl2cart(cyl_points, rad=True):
    '''
    cyl_points: [r,theta,z] (m, rad, m)
    cart_points: [x,y,z] (m, m, m)
    '''
    if type(cyl_points) != list:
        if len(cyl_points.shape) == 1:
            cyl_points = np.array([cyl_points])
    
    cart_points = []
    for i in range(0,len(cyl_points)):
        # pars
        r = cyl_points[i][0]
        theta = cyl_points[i][1]
        z = cyl_points[i][2]
        
        # cart
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        z = z
        
        cart_points.append([x,y,z])
        
    return np.array(cart_points)

def cart2cyl(cart_points):
    '''
    cart_points: [x,y,z] (m, m, m)
    cyl_points: [r,theta,z] (m, rad, m)
    '''
    if type(cart_points) != list:
        if len(cart_points.shape) == 1:
            cart_points = np.array([cart_points])
        
    cyl_points = []
    for i in range(0,len(cart_points)):
        # pars
        x = cart_points[i][0]
        y = cart_points[i][1]
        z = cart_points[i][2]
        
        # cyl
        r = math.sqrt(math.pow(x,2) + math.pow(y,2))
        theta = math.atan2(y,x)
        z = z
        
        cyl_points.append([r,theta,z])
    
    # angle is in rad    
    return np.array(cyl_points)

def generate_scan_poses(points, triggers, resolution):
        
    traces = []
    
    for i in range(0, len(points) - 1 ):

        trace_poses = []
        
        # is scanning trace
        if triggers[i]:
            # get start and end point, then interpolate
            trace = points[i:(i+2)]
            start_angle = trace[0][1]
            end_angle = trace[1][1]
            npoints = round(abs(end_angle - start_angle) / resolution)
            # [r, theta, z]
            trace_points =  np.array([np.repeat(trace[0][0], npoints, axis=0),   # r 
                            np.linspace(start_angle, end_angle, npoints, endpoint=True), # theta
                            np.repeat(trace[1][2], npoints, axis=0)]).T   # z 

            # compute poses of each point for both positioner and robot
            # [x,y,z,w,p,r,j1,2] 
            for tp in trace_points:
                xyz = cyl2cart(tp)
                # pose and cfg
                a = math.pi/2               # processing point is always at the top surface
                b =  tp[1]-math.pi/2      # virtical pose of the robot
                g = 0                       # rotation around x axis (theta pose of robot)
                j1 = -math.pi/2             # horizontal pose of positioner
                j2 = math.pi/2 - tp[1]      #  90 - theta to maintain virtical pose of robot
                trace_poses.append(np.append(xyz, [a, b, g, j1, j2]))
            
            traces.append([trace_poses, True])      # code for scanning path
        else:
            # transition path, no scan is required (only target pose it required)
            trace_points = points[i:(i+2)]
            # [x,y,z,w,p,r,j1,2]
            for tp in trace_points:
                xyz = cyl2cart(tp)
                a = math.pi/2                       # processing point is always at the top surface
                b = tp[1]-math.pi/2       # virtical pose of the robot
                g = 0                               # rotation around x axis (theta pose of robot)
                j1 = -math.pi/2                     # horizontal pose of positioner
                j2 = math.pi/2 - tp[1]              #  90 - theta to maintain virtical pose of robot
                trace_poses.append(np.append(xyz, [a, b, g, j1, j2])) 
            
            traces.append([trace_poses, False])
    
    # poses of robot and positioner are returned here with scanning trigger code
    return traces



def confirm_defects(bboxes, map, inlimits, area_thr):
    '''
    Confirm predicted defects on an image.
    bboxes: list of corrners.
    inlimits: list of inner limits. [x1,y1,x2,y2]  # in pixels
    area_thr: area size threashold, smaller will be ignore
    '''
    # input coordinate (opencv)
    # -----> x
    # |    
    # |    * (x1,y1) --
    # v    |           |
    # y    |___________* (x2, y2)
    
    
    new_bboxes = []
    for box in bboxes:
        
        b = box[0]
        # box: [x1,y1,x2,y2]
        # check defect is within processing area
        px1 = b[0]
        py1 = b[1]
        px2 = b[2]
        py2 = b[3]
        
        if (py1 >= inlimits[0]) and (py2 <= inlimits[1]):
            
            # check area size in meter^2
            # map pixels
            r = map[0,0][0]                  # for now r is same for all points
            rtheta1 = r * map[py1,px1][1]
            rtheta2 = r * map[py2,px2][1]
            z1 = map[py1,px1][2]
            z2 = map[py2,px2][2]
            
            h = abs(z2- z1)
            w = abs(rtheta2 - rtheta1)
            area = h * w
            if (area >= area_thr):
                new_bboxes.append(b) 
    
    
    return new_bboxes

def plot_surface(surface_points):

    # visulize point cloud using open3d
    point_cloud = cyl2cart(surface_points.reshape(-1,3))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    opt = vis.get_render_option()
    opt.show_coordinate_frame = True
    vis.run()
    vis.destroy_window()
