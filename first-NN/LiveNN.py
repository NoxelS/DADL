#!/usr/bin/env python3
#
# 08.12.2022 Chr. Hoffmann 
#
# Requirements:
# - pip install opencv-contrib-python opencv-python
# - pip install imageio 
# (evtl. via package-manager)
########################################################################  


########################################################################  
import cv2			# pip3 install opencv-python
import numpy as np 
from scipy import ndimage
from nn import Network
########################################################################  


########################################################################  
# Parameter
win_width, win_height	= 500,150
# globals for mouse_callback
pen_down, but3_down, old_x , old_y = False, False, 0, 0
########################################################################  


########################################################################  
def mouse_cb(event, x, y, flags, param):
  global board, pen_down, but3_down, old_x, old_y
  but3_down = flags & 0x4
  if event == cv2.EVENT_LBUTTONDOWN:
    pen_down = True
  elif event == cv2.EVENT_MOUSEMOVE:
    if pen_down:
      cv2.line(board, (old_x,old_y), (x,y), color=255, thickness=2)
    old_x, old_y = x, y
  elif event == cv2.EVENT_LBUTTONUP:
    pen_down = False
########################################################################  


########################################################################  
# main
cv2.namedWindow('Board')
cv2.moveWindow('Board', 1000,100)
cv2.setMouseCallback('Board', mouse_cb)
board = np.zeros((win_height, win_width), np.uint8)
cv2.rectangle(board, (50,50), (100,100), color=255, thickness=1)
#cv2.imshow('Board', board)

# NN
nn = Network.load("trained_network.dat")

while True:
  key = cv2.waitKey(20) & 0xFF
  if key == 27:
    break
  elif key == ord(' '):
    cv2.rectangle(board, (50,50), (100,100), color=0, thickness=1)
    win = board[50:100, 50:100]
    win = ndimage.filters.gaussian_filter(win, sigma=1)

    data = []

    for i in range(50):
      for j in range(50):
        data.append(win[i][j])

    data = np.array(data)
    data = data.reshape((len(data), 1))
    data = data / 255.0

    prediction = nn.feedforward(data)
    max_value = np.argmax(prediction)

    # Map max value to char
    if max_value == 0:
        max_value = 'a'
    elif max_value == 1:
        max_value = 'b'
    elif max_value == 2:
        max_value = 'c'

    # Print text on board

    board = np.zeros((win_height, win_width), np.uint8)

    # Print confidence on board
    cv2.putText(board, "a: " + str(prediction[0]), (300, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255 ,0), 1, cv2.LINE_AA)
    cv2.putText(board, "b: " + str(prediction[1]), (300, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(board, "c: " + str(prediction[2]), (300, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

    # Draw filled rectangle
    for i in range(3):
      cv2.putText(board, chr(ord('a') + i), (145 + i * 30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255 ,0), 1, cv2.LINE_AA)
      cv2.rectangle(board, (165 + i * 30, 115), (140 + i * 30, 115 - int(80 * prediction[i])), color=(255,0,0), thickness=-1)

    cv2.rectangle(board, (50,50), (100,100), color=255, thickness=1)

  cv2.rectangle(board, (50,50), (100,100), color=0, thickness=1)
  win = board[50:100, 50:100]
  win = ndimage.filters.gaussian_filter(win, sigma=1)

  data = []

  for i in range(50):
    for j in range(50):
      data.append(win[i][j])

  data = np.array(data)
  data = data.reshape((len(data), 1))
  data = data / 255.0

  prediction = nn.feedforward(data)
  max_value = np.argmax(prediction)

  # Map max value to char
  if max_value == 0:
      max_value = 'a'
  elif max_value == 1:
      max_value = 'b'
  elif max_value == 2:
      max_value = 'c'

  # board = np.zeros((win_height, win_width), np.uint8)
  # Make board black on the right side
  cv2.rectangle(board, (103,0), (win_width,win_height), color=0, thickness=-1)
  if board[50:100, 50:100].any() == True:
    cv2.putText(board, "a: " + str(int(np.round(prediction[0]*100,2))) + "%", (300, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255 ,0), 1, cv2.LINE_AA)
    cv2.putText(board, "b: " + str(int(np.round(prediction[1]*100,2))) + "%", (300, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(board, "c: " + str(int(np.round(prediction[2]*100,2))) + "%", (300, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

  # Draw filled rectangle
  for i in range(3):
    cv2.putText(board, chr(ord('a') + i), (145 + i * 30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255 ,0), 1, cv2.LINE_AA)
    # check if there are any values in the rectangle 50,50 100,100
    if board[50:100, 50:100].any() == True:
      cv2.rectangle(board, (165 + i * 30, 115), (140 + i * 30, 115 - int(80 * prediction[i])), color=(255,0,0), thickness=-1)

  cv2.rectangle(board, (50,50), (100,100), color=255, thickness=1)

  cv2.imshow('Board', board)
    
cv2.destroyAllWindows()

########################################################################  

