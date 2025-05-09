# Embedded Code
This folder contains the C++ files that run on the microcontroller. Please see code for more detailed description/comments, but the main functionality is as follows:

1. The system wakes up and counts down on the screen the time until the picture for classification will be taken
2. Classification of the object will be performed by the "ClassifyLCD" function
3. The result will be displayed to the user on the LCD, setting the backlight color to correspond with the classification
4. The system will set the SNVS register to turn off all systems, waiting until the next power cycle as determined by the resetting of the power system
