# 2022_S1_CS373_AssignmentSkeleton

This repository provides a Python 3 code skeleton for the image processing assignment of CompSci 373 in Semester 1, 2022.

This assignment will require you to use what we have studied in the image processing lectures to generate a software that detects license plates in images of cars - a technology that is used routinely for example on toll roads for automatic toll pricing.

You will receive 10 marks for solving the license plate detection problem, and there will be an additional component for 5 marks, where you will extend upon the license plate detection, and write a short reflective report about your extension.


# README Comments from Ou-An Chuang (ochu761)

See requirements.txt for required libraries which can be installed (if not already) with the following:

    pip install matplotlib
    pip install opencv-python==4.5.4.60
    pip install easyocr

Note: Requires older version of OpenCV; newer version may cause runtime issues
- See https://stackoverflow.com/questions/70573780/unknown-opencv-exception-while-using-easyocr
- See Troubleshooting (below)

EasyOCR requires a detection model which will be downloaded automatically during runtime.

## Changing input image
- The input image can be changed by editing the variable 'input_filename' at the top of main()

## Console input syntax
- [key] square brackets indicates key press
- (key) round brackets indicates key entered into console

## Using extension program
- The program will first compute the bounding boxes for the licence plate as done in Main Task (progress can be seen in console)
- After this is complete, the identified plate number and other letters detected will be output in console

- A window will popup to allow the user to draw on the image to make modifications (instructions/help in console). NOTE: key presses should be done when selecting the window, rather than entering into console
- Press [ESC] escape key to finish drawing session and move onto modification options
- Modification options allows the user to either send the modified image back for detection again, or sharpen or blur the image for further editing
- The user may also choose to quit/exit the program at this stage by entering (q) into the console

# Troubleshooting
Due to (probably) different versions of OpenCV not working consistently, if the extension program still generates an error, do the following:

    pip install opencv-contrib-python
    pip uninstall opencv-contrib-python
    pip uninstall opencv-python
    pip install opencv-python==4.5.4.60

Not sure why this fixes it, but probably has something to do with clashing versions where particular functions do not work in specific versions
