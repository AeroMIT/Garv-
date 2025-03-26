# Garv-

Trackbar class syntax:

1. Download trackbars.py and move it to Python\Python310\Lib folder.
2. Import trackbar: "from trackbars import Trackbar".
3. Initialize the trackbar object: "trackbar1 = Trackbar(image, mode = 'hsv' or 'rectangle' or 'circle')"
4. Inside the while loop, obtain the modified image: "edited_image = trackbar1.get()"
5. Display the image and see the changes: "cv.imshow('window', edited_image)"
