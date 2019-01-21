import pylab
import math
import uuid
import numpy as np
from cvxpy import *
from cvxconsensus import *
from collections import defaultdict

# Adapted from https://github.com/cvxgrp/cvxpy/blob/master/examples/floor_packing.py
class Box(object):
    """ A box in a floor packing problem. """
    ASPECT_RATIO = 5.0
    def __init__(self, min_area):
        self.id = uuid.uuid4()
        self.min_area = min_area
        self.height = Variable()
        self.width = Variable()
        self.x = Variable()
        self.y = Variable()

    @property
    def position(self):
        return (np.round(self.x.value,2), np.round(self.y.value,2))

    @property
    def size(self):
        return (np.round(self.width.value,2), np.round(self.height.value,2))

    @property
    def left(self):
        return self.x

    @property
    def right(self):
        return self.x + self.width

    @property
    def bottom(self):
        return self.y

    @property
    def top(self):
        return self.y + self.height

class FloorPlan(object):
    """ A minimum perimeter floor plan. """
    MARGIN = 1.0
    ASPECT_RATIO = 5.0
    def __init__(self, boxes):
        self.boxes = boxes
        self.height = Variable()
        self.width = Variable()
        self.horizontal_orderings = []
        self.vertical_orderings = []

    @property
    def size(self):
        return (np.round(self.width.value,2), np.round(self.height.value,2))

    # Return constraints for the ordering.
    @staticmethod
    def _order(boxes, horizontal):
        if len(boxes) == 0: return
        constraints = defaultdict(list)
        curr = boxes[0]
        for box in boxes[1:]:
            if horizontal:
                constraints[box.id].append(curr.right + FloorPlan.MARGIN <= box.left)
            else:
                constraints[box.id].append(curr.top + FloorPlan.MARGIN <= box.bottom)
            curr = box
        return constraints

    # Compute minimum perimeter layout.
    def layout(self, *args, **kwargs):
        size_constrs = {}
        for box in self.boxes:
            constraints = []
            # Enforce that boxes lie in bounding box.
            constraints += [box.bottom >= FloorPlan.MARGIN,
                            box.top + FloorPlan.MARGIN <= self.height]
            constraints += [box.left >= FloorPlan.MARGIN,
                            box.right + FloorPlan.MARGIN <= self.width]
            # Enforce aspect ratios.
            constraints += [(1/box.ASPECT_RATIO)*box.height <= box.width,
                            box.width <= box.ASPECT_RATIO*box.height]
            # Enforce minimum area
            constraints += [
                geo_mean(vstack((box.width, box.height))) >= math.sqrt(box.min_area)
                ]
            size_constrs[box.id] = constraints

        # Enforce the relative ordering of the boxes.
        order_constrs = []
        for ordering in self.horizontal_orderings:
            order_constrs.append(self._order(ordering, True))
        for ordering in self.vertical_orderings:
            order_constrs.append(self._order(ordering, False))

        # Form a separate problem for each box.
        p_list = []
        for box in self.boxes:
            constraints = size_constrs[box.id]
            for constrs in order_constrs:
                constraints += constrs[box.id]
            p_list += [Problem(Minimize(0), constraints)]
        p_list += [Problem(Minimize(2*(self.height + self.width)))]
        probs = Problems(p_list)
        probs.solve(*args, **kwargs)
        return probs

    # Show the layout with matplotlib
    def show(self):
        pylab.figure(facecolor='w')
        for k in range(len(self.boxes)):
            box = self.boxes[k]
            x,y = box.position
            w,h = box.size
            pylab.fill([x, x, x + w, x + w],
                       [y, y+h, y+h, y],
                       facecolor = '#D0D0D0')
            pylab.text(x+.5*w, y+.5*h, "%d" %(k+1))
        x,y = self.size
        pylab.axis([0, x, 0, y])
        pylab.xticks([])
        pylab.yticks([])

        pylab.show()

boxes = [Box(180), Box(80), Box(80), Box(80), Box(80)]
fp = FloorPlan(boxes)
fp.horizontal_orderings.append([boxes[0], boxes[2], boxes[4]])
fp.horizontal_orderings.append([boxes[1], boxes[2]])
fp.horizontal_orderings.append([boxes[3], boxes[4]])
fp.vertical_orderings.append([boxes[1], boxes[0], boxes[3]])
fp.vertical_orderings.append([boxes[2], boxes[3]])
fp.layout(method = "consensus", rho_init = 1.0, max_iter = 1000)
fp.show()