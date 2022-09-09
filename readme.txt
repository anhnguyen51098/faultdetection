            --- INSTRUCTION TO USE CODE ---

(1) Install all requirement library noted in "requirement.txt" file

(2) Download the model file and put it in the same directory
Import file "extract_object_module" in the main program

(3) Call the function dimension (extract_object_module.extract_image)
with the one parameter:

* image_path: path of image

(4) This module will run through 2 steps

- step (1): a rotation function will rotate the image
contain object in horizontal side, a image filename
called 'rotated_obj.jpg' will appear in folder
- step (2): using deep learning model to detect the
object in image and export the image called
'detect_obj.jpg' for checking. Then each device in
image will be extract as 'obj x.jpg', with x is the
number of object. Those object will be arrange in order
Then, a list of extracted image will be returned

*Notes: we have make "test" as an example to present
how to use the code in "extract_object_module"