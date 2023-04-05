import math as m
import numpy as np
import matplotlib.pyplot as plt
from src.helper import generate_scan_points
def Rz(theta):
  return np.matrix([[ m.cos(theta), -m.sin(theta), 0 ],
                   [ m.sin(theta), m.cos(theta) , 0 ],
                   [ 0           , 0            , 1 ]])
angle=60
p1 = np.squeeze(np.asarray(Rz(m.radians(angle)) * np.array([[-50],[25],[1]])))   #[0.06,m.radians(60),0.4]
p2 = np.squeeze(np.asarray(Rz(m.radians(angle)) * np.array([[-50],[-25],[1]])))   #[0.06,m.radians(30),0.5]
p23 = np.squeeze(np.asarray(Rz(m.radians(angle)) * np.array([[0],[-30],[1]])))
p3 = np.squeeze(np.asarray(Rz(m.radians(angle)) * np.array([[50],[-25],[1]])))  #[0.06,m.radians(60),0.7]
p4 = np.squeeze(np.asarray(Rz(m.radians(angle)) * np.array([[50],[25],[1]])))   #[0.06,m.radians(90),0.6]

p1 = np.array([p1[2],p1[0],p1[1]])
p2 = np.array([p2[2],p2[0],p2[1]])
p23 = np.array([p23[2],p23[0],p23[1]])
p3 = np.array([p3[2],p3[0],p3[1]])
p4 = np.array([p4[2],p4[0],p4[1]])

corrners = np.array([p1,p2,p23,p3,p4])

generate_scan_points(corrners,5, True)