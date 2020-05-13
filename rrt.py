"""Application to automatically keep track of PDFs."""

from collections import defaultdict
from functools import partial
import os
import sys
import time

from PySide2.QtWidgets import (QApplication, QDialog, QListWidget,
                               QListWidgetItem, QVBoxLayout, QHBoxLayout,
                               QPushButton, QAbstractItemView,
                               QSizePolicy)
from matplotlib.backends.backend_qt5agg import (FigureCanvas)
from matplotlib.figure import Figure
import numpy as np
from PySide2 import QtCore

class Tree():
    def __init__(self, root_node):
        self.nodes = [root_node]
        self.root = root_node
        
    def add_child(self, node, parent):
        self.nodes.append(node)
        node.parent = parent
        parent.children.append(node)

class Node():
    def __init__(self, pose, parent=None):
        self.pose = np.array(pose)
        self.parent = parent
        self.children = []

class GUI(QDialog):
    def __init__(self, parent=None):
        super(GUI, self).__init__(parent)

        self.sample_1_button = QPushButton('1 sample')
        self.sample_1_button.clicked.connect(partial(GUI.steps, self, 1))
        self.sample_10_button = QPushButton('10 samples')
        self.sample_10_button.clicked.connect(partial(GUI.steps, self, 10))
        self.sample_100_button = QPushButton('100 samples')
        self.sample_100_button.clicked.connect(partial(GUI.steps, self, 100))
        self.reset_button = QPushButton('Reset')
        self.reset_button.clicked.connect(self.reset)
        self.figure = Figure(dpi=85,
                             facecolor=(1, 1, 1),
                             edgecolor=(0, 0, 0))
        self.ax = self.figure.subplots()
        self.figure_canvas = FigureCanvas(self.figure)
        self.figure_canvas.setSizePolicy(QSizePolicy.Expanding, 
                                         QSizePolicy.Expanding)

        layout = QHBoxLayout()
        settings_layout = QVBoxLayout()
        settings_layout.addWidget(self.sample_1_button)
        settings_layout.addWidget(self.sample_10_button)
        settings_layout.addWidget(self.sample_100_button)
        settings_layout.addWidget(self.reset_button)
        layout.addLayout(settings_layout, 1)

        layout.addWidget(self.figure_canvas, 2)
        self.setLayout(layout)
        
        self.samples = []
        self.map = np.zeros((100,100))
        self.map[30:40,0:70] = 1
        self.map[50:70,30:40] = 1
        self.map[50:100,60:70] = 1
        self.map[10:20,50:60] = 1
        self.vertices = []
        self.start = np.array([10,10])
        self.goal = np.array([10,90])
        
        
        self.step_size = 4
        self.goal_sample_every_n_samples = 10
        
        self.reset()
        
    def reset(self):
        self.tree = Tree(Node(self.start))
        self.goal_sample_in = self.goal_sample_every_n_samples
        self.goal_reached = False
        self.goal_node = None
        
        self.update_plot(self.map)
        
    def distance(self, pose_1, pose_2):
        return np.linalg.norm((pose_1 - pose_2))
        
    def closest_node(self, pose):
        queue = [self.tree.root]
        best = None
        smallest_distance = 10000
        while len(queue):
            node = queue.pop(0)
            dist = self.distance(node.pose, pose)
            if dist < smallest_distance:
                best = node
                smallest_distance = dist
            for child in node.children:
                queue.append(child)
        
        return best, smallest_distance
    
    def collision_free(self, pose):
        index = (pose+0.5).astype(np.int)
        try:
            if self.map[index[1],index[0]] != 0:
                return False
            else:
                return True
        except:
            return False
        
    def steps(self, n):
        for i in np.arange(n):
            self.goal_sample_in -= 1
            goal_attempt = False
            sample_reached = False
            
            if self.goal_sample_in == 0:
                self.goal_sample_in = self.goal_sample_every_n_samples
                sample = self.goal
                goal_attempt = True
            else:
                sample = np.array([np.random.uniform(0,self.map.shape[0]+1), np.random.uniform(0,self.map.shape[1]+1)])
            
            
            closest_node, distance = self.closest_node(sample)
            
            if distance < self.step_size and distance > 0:
                sample_reached = True
                new_pose = sample
            elif distance > 0:
                new_pose = closest_node.pose + self.step_size*(sample - closest_node.pose) / distance
            else:
                continue
            
            if self.collision_free(new_pose):
                new_node = Node(new_pose)
                self.tree.add_child(new_node, closest_node)
                if goal_attempt and sample_reached:
                    self.goal_reached = True
                    self.goal_node = new_node

        self.update_plot(self.map, sample)

    def update_plot(self, map, sample=None):
        self.ax.clear()
        self.ax.imshow(map,cmap='gray_r')
        
        queue = [self.tree.root]
        while len(queue):
            node = queue.pop(0)
            self.ax.scatter(node.pose[0], node.pose[1], c='k', s=10, zorder=2)
            if node.parent is not None:
                self.ax.plot([node.pose[0], node.parent.pose[0]], 
                             [node.pose[1], node.parent.pose[1]],
                             c = 'k', zorder=1)
            for child in node.children:
                queue.append(child)
                
        if self.goal_reached:
            node = self.goal_node
            while node.parent is not None:
                self.ax.plot([node.pose[0], node.parent.pose[0]], 
                             [node.pose[1], node.parent.pose[1]],
                             c = 'r', zorder=1)
                node = node.parent
                
        
        self.ax.scatter(self.start[0], self.start[1], zorder=3, s=40)
        self.ax.scatter(self.goal[0], self.goal[1], zorder=3, s=40)
        if sample is not None:
            self.ax.scatter(sample[0],sample[1], zorder=3, s=25)
        
        self.ax.set_xlim(0,map.shape[0])
        self.ax.set_ylim(0,map.shape[1])
        self.figure.canvas.draw()


def main():
    """Execute the program."""
    app = QApplication(sys.argv)

    gui = GUI()
    gui.show()

    app.exec_()

if __name__ == "__main__":
    main()
