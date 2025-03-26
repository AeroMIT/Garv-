import cv2 as cv
import numpy as np
from math import *

class Trackbar:
    colors = {
        'red': (0, 0, 255),
        'green': (0, 255, 0),
        'blue': (255, 0, 0),
        'yellow': (0, 255, 255),
        'cyan': (255, 255, 0),
        'magenta': (255, 0, 255),
        'white': (255, 255, 255),
        'black': (0, 0, 0),
        'orange': (0, 165, 255),
        'gray': (128, 128, 128),
        'lime': (0, 255, 128),
        'magenta': (255, 0, 255),
        'navy': (128, 0, 0),
        'teal': (128, 128, 0),
        'maroon': (0, 0, 128),
    }

    rect_parameters = {
        'MoveX': None,
        'MoveY': None,
        'Width': None,
        'Height': None,
        'Fill': 1,
        'Outline': 50, 
        'TxtSize': 50, 
        'TxtOutline': 50, 
        'ShowLoc': 1, 
        'ShowDim': 1, 
        'ShowArea': 1
    }
    
    circle_parameters = {
        'MoveX': None,
        'MoveY': None,
        'Radius': None,
        'Outline': 50, 
        'Fill': 1,
        'TxtSize': 50, 
        'TxtOutline': 50, 
        'ShowLoc': 1, 
        'ShowDim': 1, 
        'ShowArea': 1
    }
    
    poly_parameters = {
        'Fill': 1,
        'Outline': 50, 
        'TxtSize': 50, 
        'TxtOutline': 50, 
        'ShowLoc': 1,
    }
    
    hsv_parameters = {
        'H1': 180,
        'S1': 255,
        'V1': 255,
        'H2': 180,
        'S2': 255,
        'V2': 255,
        'Dilations': 100,
        'Erosions': 100,
        'KernelSize':21,
        'Invert': 1
    }

    canny_parameters = {
        'Lower': 255, 
        'Upper': 255, 
        'Dilations': 100, 
        'Erosions': 100, 
        'KernelSize': 21
    }

    all_trackbars = []
    
    def __init__(self, source, mode, n_sides=3, window_name='Trackbar', window_size=(640, 640)):
        self.mode = mode
        self.window_name = window_name
        self.window_size = window_size
        self.n_sides = n_sides
        self.is_video = isinstance(source, cv.VideoCapture)
        Trackbar.all_trackbars.append(self)

        if self.is_video:
            self.video = source
            isTrue, frame = self.video.read()
            self.image = frame
        else:
            self.image = source

        Trackbar.circle_parameters['MoveX'] = self.image.shape[1]
        Trackbar.circle_parameters['MoveY'] = self.image.shape[0]
        Trackbar.circle_parameters['Radius'] = self.image.shape[0]

        Trackbar.rect_parameters['MoveX'] = self.image.shape[1]
        Trackbar.rect_parameters['MoveY'] = self.image.shape[0]
        Trackbar.rect_parameters['Width'] = self.image.shape[1]
        Trackbar.rect_parameters['Height'] = self.image.shape[0]
        
        cv.namedWindow(self.window_name)
        cv.resizeWindow(self.window_name, self.window_size[0], self.window_size[1])
        self._build_trackbars()

    def _nothing(self, x):
        return x

    def _create_rect_trackbar(self):
        
        for parameter, limit in Trackbar.rect_parameters.items():
            cv.createTrackbar(parameter, self.window_name, 0, limit, self._nothing)

    def _create_circle_trackbar(self):
        
        for parameter, limit in Trackbar.circle_parameters.items():
            cv.createTrackbar(parameter, self.window_name, 0, limit, self._nothing)
            
    def _create_polygon_trackbar(self):
        
        for i in range(self.n_sides):
            cv.createTrackbar(f'X{i+1}', self.window_name, 0, self.image.shape[1], self._nothing)
            cv.createTrackbar(f'Y{i+1}', self.window_name, 0, self.image.shape[0], self._nothing)
            
        for parameter, limit, in Trackbar.poly_parameters.items():
            cv.createTrackbar(parameter, self.window_name, 0, limit, self._nothing)

    def _create_hsv_mask_trackbar(self):
        
        for parameter, limit in Trackbar.hsv_parameters.items():
            cv.createTrackbar(parameter, self.window_name, 0, limit, self._nothing)

    def _create_threshold_trackbar(self):
        
        pass

    def _create_edge_detection_trackbar(self):
        
        pass

    def _create_canny_trackbar(self):
        
        for parameter, limit in Trackbar.canny_parameters.items():
            cv.createTrackbar(parameter, self.window_name, 0, limit, self._nothing)

    def _create_resize_trackbar(self):
        
        cv.createTrackbar('Scale', self.window_name, 1, 100, self._nothing)

    def _build_trackbars(self):

        match self.mode:
            case 'rectangle': self._create_rect_trackbar()
            case 'circle': self._create_circle_trackbar()
            case 'mask': self._create_hsv_mask_trackbar()
            case 'polygon': self._create_polygon_trackbar()
            case 'threshold': self._create_threshold_trackbar()
            case 'adaptive': self._create_adathreshold_trackbar()
            case 'edge': self._create_edge_detection_trackbar()
            case 'canny': self._create_canny_trackbar()
            case 'resize': self._create_resize_trackbar()
            case _: raise ValueError("No such mode.")

    def get(self, shapeColor='green', textColor='green'):
        
        if self.is_video:
            isTrue, frame = self.video.read()
            if not isTrue:
                self.video.set(cv.CAP_PROP_POS_FRAMES, 0)
                isTrue, frame = self.video.read()
                
            self.image = frame

        self.hsv = cv.cvtColor(self.image, cv.COLOR_BGR2HSV)
        self.gray = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
        self.copy = self.image.copy()
        self.result = None

        try:
            match self.mode:
                case 'rectangle': self._get_rect_trackbar(shapeColor, textColor)
                case 'circle': self._get_circle_trackbar(shapeColor, textColor)
                case 'polygon': self._get_polygon_trackbar(shapeColor, textColor)
                case 'mask': self._get_hsv_trackbar()
                case 'threshold': self._get_threshold_trackbar()
                case 'adaptive': self._get_adathreshold_trackbar()
                case 'edge': self._get_edge_detection_trackbar()
                case 'canny': self._get_canny_trackbar()
                case 'resize': self._get_resize_trackbar()
                
        finally:
            return self.result

    def _get_rect_trackbar(self, shapeColor, textColor):
        
        x, y = cv.getTrackbarPos('MoveX', self.window_name), cv.getTrackbarPos('MoveY', self.window_name)
        w, h = cv.getTrackbarPos('Width', self.window_name), cv.getTrackbarPos('Height', self.window_name)
        
        fill = cv.getTrackbarPos('Fill', self.window_name)
        shapeThickness = max(1, cv.getTrackbarPos('Outline', self.window_name))
        textThickness = max(1, cv.getTrackbarPos('TxtOutline', self.window_name))
        textSize = max(1, cv.getTrackbarPos('TxtSize', self.window_name))
        
        showCoord = cv.getTrackbarPos('ShowLoc', self.window_name)
        showDim = cv.getTrackbarPos('ShowDim', self.window_name)
        showArea = cv.getTrackbarPos('ShowArea', self.window_name)

        cv.rectangle(self.copy, (x,y), (x+w,y+h), color = Trackbar.colors[shapeColor], thickness = -1 if fill else shapeThickness)
   
        if showCoord:
            cv.putText(self.copy, f'({x},{y})',(x-50,y-35),cv.FONT_HERSHEY_SIMPLEX, textSize, Trackbar.colors[textColor], thickness=textThickness)
        if showDim:
            cv.putText(self.copy, f'{w}',((x+x+w)//2,y-5),cv.FONT_HERSHEY_SIMPLEX, textSize, Trackbar.colors[textColor], thickness=textThickness)
            cv.putText(self.copy, f'{h}',((x+w+5),(y+y+h)//2),cv.FONT_HERSHEY_SIMPLEX, textSize, Trackbar.colors[textColor], thickness=textThickness)   
        if showArea:
            cv.putText(self.copy, f'{w*h}',(x-50,y+h+35),cv.FONT_HERSHEY_SIMPLEX, textSize, Trackbar.colors[textColor], thickness=textThickness)

        self.result = self.copy
            
    def _get_circle_trackbar(self, shapeColor, textColor):
        
        x, y, r = cv.getTrackbarPos('MoveX', self.window_name), cv.getTrackbarPos('MoveY', self.window_name), cv.getTrackbarPos('Radius', self.window_name)
        centre = (x,y)
        
        fill = cv.getTrackbarPos('Fill', self.window_name)
        shapeThickness = max(1, cv.getTrackbarPos('Outline', self.window_name))
        textThickness = max(1, cv.getTrackbarPos('TxtOutline', self.window_name))
        textSize = max(1, cv.getTrackbarPos('TxtSize', self.window_name))

        showCoord = cv.getTrackbarPos('ShowLoc', self.window_name)
        showDim = cv.getTrackbarPos('ShowDim', self.window_name)
        showArea = cv.getTrackbarPos('ShowArea', self.window_name)
        
        cv.circle(self.copy, centre, r, Trackbar.colors[shapeColor],thickness = -1 if fill else shapeThickness)
                
        if showCoord:
            cv.putText(self.copy, f'({x},{y})', (x,y), cv.FONT_HERSHEY_SIMPLEX, textSize, Trackbar.colors[textColor], thickness=textThickness)
        if showDim:
            cv.putText(self.copy, f'{r}', (x+30,y-r-5), cv.FONT_HERSHEY_SIMPLEX, textSize, Trackbar.colors[textColor], thickness=textThickness)
        if showArea:
            cv.putText(self.copy, f'{int(pi*r*r)}', (x+r+5,y), cv.FONT_HERSHEY_SIMPLEX, textSize, Trackbar.colors[textColor], thickness=textThickness)

        self.result = self.copy
        
    def _get_polygon_trackbar(self, shapeColor, textColor):
        
        points = np.array([[cv.getTrackbarPos(f'X{i}', self.window_name), cv.getTrackbarPos(f'Y{i}', self.window_name)]
                             for i in range(1, self.n_sides + 1)], dtype=np.int32)

        shapeThickness = max(1, cv.getTrackbarPos('Outline', self.window_name))
        textThickness = max(1, cv.getTrackbarPos('TxtOutline', self.window_name))
        textSize = max(1, cv.getTrackbarPos('TxtSize', self.window_name))
        
        if not cv.getTrackbarPos('Fill', self.window_name):
            cv.polylines(self.copy, [points], isClosed = True, color = Trackbar.colors[shapeColor], thickness = shapeThickness)
        else:
            cv.fillPoly(self.copy, [points], color = Trackbar.colors[shapeColor])

        if showCoord:
            for i in range(1,self.n_sides+1):
                cv.putText(self.copy, f'(x{i},y{i})', points[i-1], cv.FONT_HERSHEY_SIMPLEX, textSize, Trackbar.colors[textColor], thickness=textThickness)

        self.result = self.copy

    def _get_hsv_trackbar(self):
        
        H1, S1, V1 = cv.getTrackbarPos('H1', self.window_name), cv.getTrackbarPos('S1', self.window_name), cv.getTrackbarPos('V1', self.window_name) 
        H2, S2, V2 = cv.getTrackbarPos('H2', self.window_name), cv.getTrackbarPos('S2', self.window_name), cv.getTrackbarPos('V2', self.window_name)
        i, j = cv.getTrackbarPos('Dilations', self.window_name), cv.getTrackbarPos('Erosions', self.window_name)
        n = cv.getTrackbarPos('KernelSize', self.window_name)
        n = max(3, n if n % 2 else n + 1)
        inverse = cv.getTrackbarPos('Invert', self.window_name)
        
        kernel = (n, n)
        lower = np.array([H1, S1, V1])
        upper = np.array([H2, S2, V2])
        
        mask = cv.inRange(self.hsv, lower, upper)
        mask = cv.bitwise_not(mask) if inverse else mask

        mask = cv.erode(mask, kernel, iterations=j)
        mask = cv.dilate(mask, kernel, iterations=i)

        mask_result = cv.bitwise_and(self.copy, self.copy, mask = mask)

        self.result = mask, mask_result

    def _get_threshold_trackbar(self):
        pass

    def _get_adathreshold_trackbar(self):
        pass

    def _get_edge_detection_trackbar(self):
        pass

    def _get_canny_trackbar(self):
        
        lower, upper = cv.getTrackbarPos('Lower', self.window_name), cv.getTrackbarPos('Upper', self.window_name)
        i, j = cv.getTrackbarPos('Dilations', self.window_name), cv.getTrackbarPos('Erosions', self.window_name)
        n = cv.getTrackbarPos('KernelSize', self.window_name)
        n = max(3, n if n % 2 else n + 1)
        
        self.result = cv.Canny(self.gray, lower, upper, n)
        self.result = cv.erode(result, (n,n), iterations=j)
        self.result = cv.dilate(result, (n,n), iterations=i)

    def _get_resize_trackbar(self):
        
        scale = max(1, cv.getTrackbarPos('Scale', self.window_name))
        interpolation =  cv.INTER_AREA
    
        width = int(self.copy.shape[1] * scale/100) 
        length = int(self.copy.shape[0] * scale/100)
        dimensions = (width, length)
        self.result = cv.resize(self.copy, dimensions, interpolation)
    
    def close(self):
        cv.destroyWindow(self.window_name)
