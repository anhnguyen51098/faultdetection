#import dimension_module
import faultdetection_m3.utils as ut

image_path = r"obj 1.jpg"

# D, W, A, B = dimension_module.check_dimension(image_path, use_dl=False)
# print("D = ", D)
# print("W = ", W)
# print("A = ", A)
# print("B = ", B)

a, b = ut.threshold(image_path)

print(a)
print(b)