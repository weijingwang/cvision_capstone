from PIL import Image
from rembg import remove
import numpy as np


i_path = "test_car_street.png"
o_path = "output.png"

input = Image.open(i_path)
output = remove(input)
output.save(o_path)

out_arr = np.array(output.convert("L"))
print(out_arr.shape)

out_arr[out_arr > 0] = 255

Image.fromarray(out_arr).show()

bit_map = (out_arr/255).astype(int)
print(bit_map)

