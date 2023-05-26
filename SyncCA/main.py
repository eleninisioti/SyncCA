import numpy as np
import scipy.signal
from utils import VideoWriter
import random
import copy
from matplotlib import animation
import matplotlib.pyplot as plt
import os
from moviepy.editor import *
from PIL import Image
import glob
import moviepy.editor as mp
import math


freq = 10

def life_step_1(X):
    """Game of life step using generator expressions"""
    nbrs_count = sum(np.roll(np.roll(X, i, 0), j, 1)
                     for i in (-1, 0, 1) for j in (-1, 0, 1)
                     if (i != 0 or j != 0))
    return (nbrs_count == 3) | (X & (nbrs_count == 2))


def life_step_2(X):
    """Game of life step using scipy tools"""
    from scipy.signal import convolve2d
    nbrs_count = convolve2d(X, np.ones((3, 3)), mode='same', boundary='wrap') - X
    return (nbrs_count == 3) | (X & (nbrs_count == 2))

def life_step_fireflies(X, r_c, e, l, osc_locs):
    """
    X: array
    r_c: radius of coupling
    e: strength of coupling
    """
    # find oscillator locations

    N_c = int(len(osc_locs)*r_c) # radius of coupling
    print("new step ", str(int(N_c)))
    for loc in osc_locs:
        sum = 0
        for x in range(-int(N_c), int(N_c)+1):
            for y in range(-int(N_c), int(N_c)+1):
                new_loc = ((loc[0] +x)%np.shape(X)[0], (loc[1] +y)%np.shape(X)[1])

                if (new_loc in osc_locs) and new_loc!=loc:

                    sum += np.abs(X[loc] - X[new_loc[0], new_loc[1]])
        new_value = (X[loc] + e/(2*N_c)*sum)%l
        print(loc, new_value, sum)
        X[loc] = new_value
    return X




life_step = life_step_fireflies


def flash(x, frame):
    freq = 100
    return (np.sin(2 * math.pi * freq * frame + x) + 1)



# define vectorized sigmoid
flash_v = np.vectorize(flash)

def life_animation(X, r, e, l,osc_locs, dpi=10, frames=10, interval=300, mode='loop', filename="temp.mp4"):
    """Produce a Game of Life Animation

    Parameters
    ----------
    X : array_like
        a two-dimensional numpy array showing the game board
    dpi : integer
        the number of dots per inch in the resulting animation.
        This controls the size of the game board on the screen
    frames : integer
        The number of frames to compute for the animation
    interval : float
        The time interval (in milliseconds) between frames
    mode : string
        The default mode of the animation.  Options are ['loop'|'once'|'reflect']
    """
    X = np.asarray(X)
    assert X.ndim == 2
    #X = X.astype(bool)

    X_blank = np.zeros_like(X)
    figsize = (X.shape[1] * 1. / dpi, X.shape[0] * 1. / dpi)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1], xticks=[], yticks=[], frameon=False)
    im = ax.imshow(X, cmap=plt.cm.YlOrRd, interpolation='nearest')
    im.set_clim(-0.05, 1)  # Make background gray

    # initialization function: plot the background of each frame
    def init():
        im.set_data(X_blank)
        return (im,)

    # animation function.  This is called sequentially
    def animate(i):
        animate.X_temp = copy.copy(animate.X)
        animate.X_temp *= (255.0 / animate.X_temp.max())
        #im.set_data(animate.X_temp)
        animate.X = life_step(animate.X, r, e,l, osc_locs)
        return (im,animate.X)
    from PIL import Image
    from matplotlib import cm

    save_dir = "images/" + filename
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for frame in range(frames):
        animate.X = X
        im, X = animate("x")
        animate.X = X

        #im, X = Image.fromarray(X)
        #temp = np.uint8(cm.YlOrRd(X) )

        # X holds the phase values. need to convert to

        phase_x = flash_v(X, frame)
        #phase_x = (phase_x - phase_x.min())/(phase_x.max() - phase_x.min())
        #phase_x = list(map(flash, X, [frame]*len(X)))
        rgb_array = np.zeros((X.shape[0], X.shape[1], 3), dtype=np.uint8)
        print("phases", phase_x[5,4], phase_x[5,6])
        phase_x = phase_x*(255/2)
        phase_x = phase_x.astype(np.uint8)
        rgb_array[:,:,0] = phase_x
        rgb_array[:, :, 1] = phase_x
        rows, cols = np.nonzero(phase_x)
        all_locs = [(row, cols[idx]) for idx, row in enumerate(rows)]
        not_osc_locs = [el for el in all_locs if el not in osc_locs]
        for el in not_osc_locs:
            rgb_array[el[0],el[1],:] = 0
        #rgb_array[:, :, 1] = X
        #rgb_array[:, :, 2] = X
        rgb_array = np.repeat(rgb_array, 10, axis=0)
        rgb_array = np.repeat(rgb_array, 10, axis=1)

        im =Image.fromarray(rgb_array)

        im.save(save_dir + "/gen_" + str(frame) + ".png")
        plt.clf()

    #anim = animation.FuncAnimation(fig, animate, init_func=init,
    #                               frames=frames, interval=interval)
    #writervideo = animation.FFMpegWriter(fps=1)
    #anim.save(filename, writer=writervideo)
    plt.close()

    # print anim_to_html(anim)

def merge_videos(directory, num_gens):
    frames = []
    for gen in range(num_gens):
        frames.append(Image.open("images/" + directory+ "/gen_"+ str(gen) + ".png"))
    #frames = [Image.open(image) for image in glob.glob(f"{'images/' + directory}/*.png")]
    frame_one = frames[0]
    frame_one.save("images/" + directory + "/fireflies.gif", format="GIF", append_images=frames,
               save_all=True, duration=num_gens, loop=0)
    clip = mp.VideoFileClip("images/" + directory + "/fireflies.gif")
    clip.write_videofile("images/" + directory + "/fireflies.mp4")



def fireflies():
    # initial setup
    W = 10
    L = 10
    N = 20
    l = 30 # TODO
    r = 0.3 # radius of coupling (<0.5)
    e = 0.5 # strength of coupling
    np.random.seed(0)
    world = np.zeros(shape=(W, L), dtype=float)
    random_x = random.choices(range(W), k=N)
    random_y = random.choices(range(L), k=N)
    random_locs = [(el, random_y[idx]) for idx, el in enumerate(random_x)]
    for loc in random_locs:
        world[loc] = random.randrange(l)


    rows, cols = np.nonzero(world)
    osc_locs = [(row, cols[idx]) for idx, row in enumerate(rows)]
    #r = np.random.random((10, 20))
    #X[10:20, 10:30] = (r > 0.75)
    life_animation(world, r, e,l, osc_locs,dpi=300, frames=40, mode='once', filename="fireflies.mp4")


def fireflies_medium():
    # initial setup
    W = 100
    L = 100
    N = 100
    l = 255 # TODO
    r = 0.5 # radius of coupling (<0.5)
    e = 0.5 # strength of coupling
    gens = 2000
    np.random.seed(0)
    world = np.zeros(shape=(W, L), dtype=float)

    world = np.zeros(shape=(W, L), dtype=float)
    random_x = random.choices(range(W), k=N)
    random_y = random.choices(range(L), k=N)
    random_locs = [(el, random_y[idx]) for idx, el in enumerate(random_x)]
    for loc in random_locs:
        world[loc] = random.randrange(l)

    rows, cols = np.nonzero(world)
    osc_locs = [(row, cols[idx]) for idx, row in enumerate(rows)]


    #r = np.random.random((10, 20))
    #X[10:20, 10:30] = (r > 0.75)
    life_animation(world, r, e,l,osc_locs, dpi=300, frames=gens, mode='once', filename="fireflies_medium")
    merge_videos("fireflies_medium", gens)


def fireflies_small():
    # initial setup
    W = 10 #width
    L = 10 #length
    N = 2 # number of agents
    l = 255 # TODO
    r = 1 # radius of coupling (<0.5)
    e = 0.3 # strength of coupling
    gens=1000
    np.random.seed(0)
    world = np.zeros(shape=(W, L), dtype=float)

    world[int(W/2),int(L/2)-1] = random.randrange(l)
    world[int(W/2), int(L/2) + 1] = random.randrange(l)

    rows, cols = np.nonzero(world)
    osc_locs = [(row, cols[idx]) for idx, row in enumerate(rows)]


    #r = np.random.random((10, 20))
    #X[10:20, 10:30] = (r > 0.75)
    life_animation(world, r, e,l,osc_locs, dpi=300, frames=gens, mode='once', filename="fireflies_small")
    merge_videos("fireflies_small", gens)


def example():
    np.random.seed(0)
    X = np.zeros((30, 40), dtype=bool)
    r = np.random.random((10, 20))
    X[10:20, 10:30] = (r > 0.75)
    life_animation(X, dpi=10, frames=40, mode='once', filename="example.mp4")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #example()
    fireflies_small()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
