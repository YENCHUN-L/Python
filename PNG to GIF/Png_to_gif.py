import os
import imageio

png_dir = 'C:/Users/yliu10/Desktop/png' #png folder path
images = []
for file_name in os.listdir(png_dir):
    if file_name.endswith('.png'):
        file_path = os.path.join(png_dir, file_name)
        images.append(imageio.imread(file_path))
imageio.mimsave('C:/Users/yliu10/Desktop/png/0.gif', images, duration=1) #save path and file name, duration = speed in second
