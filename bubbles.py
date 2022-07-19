import cv2
from math import sqrt
import numpy as np
import mediapipe as mp
from PIL import ImageDraw, Image, ImageFont

## Global ======

BUB_MARGIN = 10
TAIL_POS_FILTER_THRESHOLD = 3

## Functions ======

def find_best_bubble_param(face_landmarks, im_size):
    """
    Returns the best possible start position, box size and tail behavior
    """
    w, h = im_size

    # mouth
    xm = face_landmarks.landmark[0].x
    ym = face_landmarks.landmark[0].y

    if ym > 0.7 :
        if xm < 0.5:
            pos = (int(0.4*w), int(0.1*h))
            size = (int(0.5*w), int((ym - 0.5)*h))
            tail_end = (int(xm*w+30), int(ym*h))
            tail_off = -0.5
        else:
            pos = (int(0.1*w), int(0.1*h))
            size = (int(0.6*w), int((ym - 0.5)*h))
            tail_end = (int(xm*w-30), int(ym*h))
            tail_off = 0.5

    else:
        if xm < 0.5 :
            pos = (int((xm+0.2)*w), int(0.1*h))
            size = (int((0.7 - xm)*w), int((0.8*h)))
            tail_end = (int(xm*w+30), int(ym*h))
            tail_off = -0.3
        else:
            pos = (int(0.1*w), int(0.1*h))
            size = (int((xm-0.3)*w), int((0.8*h)))
            tail_end = (int(xm*w-30), int(ym*h))
            tail_off = 0.3

    return pos, size, tail_end, tail_off

def rounded_rectangle(src, top_left, bottom_right, radius, color=255, thickness=1, line_type=cv2.LINE_AA):

    #  corners:
    #  p1 - p2
    #  |     |
    #  p4 - p3

    p1 = top_left
    p2 = (bottom_right[0], top_left[1])
    p3 = (bottom_right[0], bottom_right[1])
    p4 = (top_left[0], bottom_right[1])

    height = min(abs(bottom_right[0] - top_left[0]), abs(bottom_right[1] - top_left[1]))

    if radius > height//2:
        radius = height//2

    corner_radius = int(radius)

    if thickness < 0:

        #big rect
        top_left_main_rect = (int(p1[0] + corner_radius), int(p1[1]))
        bottom_right_main_rect = (int(p3[0] - corner_radius), int(p3[1]))

        top_left_rect_left = (p1[0], p1[1] + corner_radius)
        bottom_right_rect_left = (p4[0] + corner_radius, p4[1] - corner_radius)

        top_left_rect_right = (p2[0] - corner_radius, p2[1] + corner_radius)
        bottom_right_rect_right = (p3[0], p3[1] - corner_radius)

        all_rects = [
        [top_left_main_rect, bottom_right_main_rect], 
        [top_left_rect_left, bottom_right_rect_left], 
        [top_left_rect_right, bottom_right_rect_right]]

        [cv2.rectangle(src, rect[0], rect[1], color, thickness) for rect in all_rects]

    # draw straight lines
    cv2.line(src, (p1[0] + corner_radius, p1[1]), (p2[0] - corner_radius, p2[1]), color, abs(thickness), line_type)
    cv2.line(src, (p2[0], p2[1] + corner_radius), (p3[0], p3[1] - corner_radius), color, abs(thickness), line_type)
    cv2.line(src, (p3[0] - corner_radius, p4[1]), (p4[0] + corner_radius, p3[1]), color, abs(thickness), line_type)
    cv2.line(src, (p4[0], p4[1] - corner_radius), (p1[0], p1[1] + corner_radius), color, abs(thickness), line_type)

    # draw arcs
    cv2.ellipse(src, (p1[0] + corner_radius, p1[1] + corner_radius), (corner_radius, corner_radius), 180.0, 0, 90, color ,thickness, line_type)
    cv2.ellipse(src, (p2[0] - corner_radius, p2[1] + corner_radius), (corner_radius, corner_radius), 270.0, 0, 90, color , thickness, line_type)
    cv2.ellipse(src, (p3[0] - corner_radius, p3[1] - corner_radius), (corner_radius, corner_radius), 0.0, 0, 90,   color , thickness, line_type)
    cv2.ellipse(src, (p4[0] + corner_radius, p4[1] - corner_radius), (corner_radius, corner_radius), 90.0, 0, 90,  color , thickness, line_type)

    return src

def norm2(pt1, pt2):
    (x1, y1) = pt1
    (x2, y2) = pt2

    return sqrt((x1 - x2)**2 + (y1-y2)**2)

## Classes ======

class ml_text:
    def __init__(self, text_lines, text_gap = None, pil_font = None, text_color = (255, 255, 255)) -> None:
        self.lines = text_lines
        self.font = pil_font
        self.color = text_color

        if pil_font is None:
            self.font = ImageFont.truetype(r'C:\Windows\Fonts\msgothic.ttc', 20)
        else:
            self.font = pil_font 

        # Default gap between lines
        if text_gap is None:
            text_size = self.font.getsize(text_lines[0])
            text_gap = text_size[1] + 5
        self.gap = text_gap
    
    def numLines(self):
        return len(self.lines)

    def lastLine(self):
        if self.numLines() > 0:
            return self.lines[self.numLines()-1]
        else:
            return ''

    def newLine(self, text_string):
        self.lines.append(text_string)

    def addToLastLine(self, text_string):
        if self.numLines() > 0:
            self.lines[self.numLines()-1] = self.lastLine() + text_string
        else:
            self.newLine(text_string)

    def addWord(self, word):
        if self.numLines() > 0:
            self.addToLastLine(' ' + word)
        else: self.addToLastLine(word)

    def getLineHeight(self):
        line_sz = self.font.getsize(self.lines[0])

        return line_sz[1]

    def getBoundingBox(self):
        width = 0

        # Find line with max width
        for line in self.lines:
            line_sz = self.font.getsize(line)
            if line_sz[0] > width:
                width = line_sz[0]

        # find height
        height = (len(self.lines)) * self.gap

        return (width, height)

    def draw(self, src,  pos):
        """
        Draws the content of the lines one image "src" at the specified position
        pos is (x_top_left_corner, y_top_left_corner)
        """
        x0, y0 = pos
        pil_img=Image.fromarray(src)
        draw = ImageDraw.Draw(pil_img)
        for i, line in enumerate(self.lines):
            y = y0 + i*(self.gap)
            draw.text((x0, y), line, fill=self.color, font=self.font)

        open_cv_image = np.array(pil_img) 
        # Convert RGB to BGR 
        #open_cv_image = open_cv_image[:, :, ::-1].copy()

        return open_cv_image


class speech_bubble:
    def __init__(self, ml_text, txt_pos, max_txt_sz = None, tail_end = None, tail_hor_offset = 0, bg_color = (255, 255, 255), bub_radius = 7, border_thickness = 2, angry=False) -> None:
        self.text = ml_text
        self.pos = txt_pos
        self.tail = tail_end
        self.color = bg_color
        self.radius = bub_radius
        self.thickness = border_thickness
        self.is_angry = angry
        self.max_sz = max_txt_sz

        self.stay_cnt = 0

        if tail_hor_offset > 1:
            self.tail_offset = 1
            print("Warning, tail offset should be between -1.0 and 1.0, clipped to nearest valid value")
        elif tail_hor_offset < -1:
            self.tail_offset = -1
            print("Warning, tail offset should be between -1.0 and 1.0, clipped to nearest valid value")
        else:
            self.tail_offset = tail_hor_offset

        self.start_x = txt_pos[0] - BUB_MARGIN
        self.start_y = txt_pos[1] - BUB_MARGIN

        self.end_x = txt_pos[0] + BUB_MARGIN + ml_text.getBoundingBox()[0]
        self.end_y = txt_pos[1] + BUB_MARGIN + ml_text.getBoundingBox()[1]

        if tail_end is None:
            self.tail = txt_pos

    def compute_anchor_pts(self):
        """
        Computes points used to draw the bubble's tail
        """
        (xa, ya) = self.tail
        
        # Bubble's center
        xc = (self.start_x + self.end_x) //2
        yc = (self.start_y + self.end_y) //2

        center = (xc, yc)

        # Factor for anchor width
        s = min(abs(self.end_x - self.start_x), abs(self.end_y - self.start_y))//4 

        # Horizontal offset of the tail's start
        xc = int(xc + self.tail_offset*((self.start_x - self.end_x)//2 - 2*s))
        center = (xc, yc)

        # Compute 3 pts
        pts = np.array([[xa, ya], 
            [xc-s*(ya-yc)//norm2(self.tail, center), yc-s*(xc-xa)//norm2(self.tail, center)],
            [xc+s*(ya-yc)//norm2(self.tail, center), yc+s*(xc-xa)//norm2(self.tail, center)]])

        # Some magic and type conversion
        pts = pts.reshape((-1, 1, 2))
        return pts.astype(int)

    def compute_angry_spikes(self):
        """
        Computes points used to draw the spikes of angry bubbles
        """
        # Bubble's center
        xc = (self.start_x + self.end_x) //2
        yc = (self.start_y + self.end_y) //2

        (x1, y1) = (self.start_x, self.start_y)
        (x2, y2) = (self.end_x, self.end_y)

        # Create 8*3pts
        pts_list = [np.array([[x1 + 1/10*(x2-x1), yc], [xc, yc], [x1 + 1/12*(x2-x1), y1 -  1/8*(y2-y1)]]),
        np.array([[x2 - 1/8*(x2-x1), yc], [xc, yc], [x2 - 1/10*(x2-x1), y1 - 1/8*(y2-y1)]]),
        np.array([[x2 - 1/10*(x2-x1), yc], [xc, yc], [x2 - 1/12*(x2-x1), y2 + 1/8*(y2-y1)]]),
        np.array([[x1 + 1/8*(x2-x1), yc], [xc, yc], [x1 + 1/10*(x2-x1), y2 + 1/8*(y2-y1)]]),

        np.array([[xc, y1 + 1/10*(y2-y1)], [xc, yc], [x1-1/8*(x2-x1), y1 + 1/6*(y2-y1)]]),
        np.array([[xc, y1 + 1/8*(y2-y1)], [xc, yc], [x2+1/8*(x2-x1), y1 + 1/6*(y2-y1)]]),
        np.array([[xc, y2 - 1/10*(y2-y1)], [xc, yc], [x2+1/8*(x2-x1), y2 - 1/6*(y2-y1)]]),
        np.array([[xc, y2 - 1/8*(y2-y1)], [xc, yc], [x1-1/8*(x2-x1), y2 - 1/6*(y2-y1)]])
        ]

        # Some magic
        for i, pts in enumerate(pts_list):
            pts_list[i] = pts.reshape((-1, 1, 2)).astype(int)
        return pts_list

    def draw(self, src):

        if self.text.lines:
            # Draw anchor (and spikes if needed)
            anchor_pts = self.compute_anchor_pts()
            cv2.polylines(src, [anchor_pts], isClosed=True, color=(0, 0, 0), thickness=self.thickness*3, lineType=cv2.LINE_AA)

            if self.is_angry:
                spikes_pts = self.compute_angry_spikes()
                for spike in spikes_pts:
                    cv2.polylines(src, [spike], isClosed=True, color=(0, 0, 0), thickness=self.thickness*3, lineType=cv2.LINE_AA)

            # Draw bubble body
            src = rounded_rectangle(src, (self.start_x, self.start_y), (self.end_x, self.end_y), self.radius, self.color, -1, cv2.LINE_AA)
            src = rounded_rectangle(src, (self.start_x, self.start_y), (self.end_x, self.end_y), self.radius, (0,0,0), self.thickness, cv2.LINE_AA)

            # Fill anchor (and spikes if needed)
            cv2.fillPoly(src, [anchor_pts], color=self.color)
            if self.is_angry:
                for spike in spikes_pts:
                    cv2.fillPoly(src, [spike], color=self.color)

            # Draw text
            src = self.text.draw(src, self.pos)
        return src

    def updateSize(self):
        self.start_x = self.pos[0] - BUB_MARGIN
        self.start_y = self.pos[1] - BUB_MARGIN

        self.end_x = self.pos[0] + BUB_MARGIN + self.text.getBoundingBox()[0]
        self.end_y = self.pos[1] + BUB_MARGIN + self.text.getBoundingBox()[1]

    def updateText(self, ml_text):
        self.text = ml_text
        self.updateSize()

    def updateTail(self, tail_end_pos):
        if norm2(self.tail, tail_end_pos) > TAIL_POS_FILTER_THRESHOLD:
            self.tail = tail_end_pos

    def addWord(self, word_string):
        size = self.text.font.getsize(self.text.lastLine() + ' ' + word_string)
        if self.max_sz is not None and self.text.lines:
            if size[0] < self.max_sz[0]:
                self.text.addWord(word_string)
            elif self.text.getBoundingBox()[1] + self.text.getLineHeight() < self.max_sz[1]:
                self.text.newLine(word_string)
            else:
                self.text.lines = [word_string]
        else:
            self.text.addWord(word_string)

        self.stay_cnt = 0

        #self.updateSize()
    
    def stay(self):
        self.stay_cnt +=1

    def update(self, words_list, face_landmarks, emotion, im_size, stay_frames):
        pos, size, tail_end, tail_offset = find_best_bubble_param(face_landmarks, im_size)

        if words_list:
            self.text.lines = []
            for word in words_list:
                self.addWord(word)
        elif self.stay_cnt < stay_frames and self.text.lines:
            self.stay()
        else:
            self.text = ml_text([], self.text.gap, self.text.font, self.text.color)
            self.pos = pos
            self.max_sz = size
            self.is_angry = emotion
            self.tail = tail_end
            self.tail_offset = tail_offset

        self.updateSize()
        self.updateTail(tail_end)
