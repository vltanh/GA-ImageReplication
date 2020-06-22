import os
import imageio

path = 'output'
images = [  imageio.imread(f'{path}/{x}') 
            for i, x in enumerate(sorted(os.listdir(path))) ]
imageio.mimsave('output.gif', images, duration=0.001)