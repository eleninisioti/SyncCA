
def fireflies():
    import numpy as np
    frames = np.random.randint(256, size=[20, 64, 64, 1], dtype=np.uint8)  # YOUR DATA HERE

    # save it as a gif
    from moviepy.editor import ImageSequenceClip
    clip = ImageSequenceClip(list(frames), fps=20)
    clip.write_gif('test.gif', fps=20)

    # initialize
    timesteps = 200 # timesteps of simulation
    vid = VideoWriter("media.mp4", 20.0)
    W = 200 # world width
    L = 200 # world length
    N = 100 # number of oscillators
    l = 50 # maximum phase value
    world = np.random.rand(W, L)

    world = np.zeros(shape=(W, L))
    random_x = random.choices(range(W), k=100)
    random_y = random.choices(range(L), k=100)
    random_locs = [(el, random_y[idx]) for idx, el in enumerate(random_x)]
    for loc in random_locs:
        world[loc] = random.randrange(l)


    for t in range(timesteps):
        world = np.random.rand(W, L)

        vid.add(world)
    vid.close()



def fireflies_old():
    print("Fireflies simulation")
    size = 64
    T = 10
    ''' define kernel radius '''
    R = 5
    np.random.seed(0)
    global A
    A = np.random.rand(size, size)
    ''' larger rectangular kernel '''
    # K = np.asarray([[1,1,1], [1,0,1], [1,1,1]])
    K = np.ones((2 * R + 1, 2 * R + 1));
    K[R, R] = 0
    K = K / np.sum(K)

    def growth(U):
        return 0 + ((U >= 0.12) & (U <= 0.15)) - ((U < 0.12) | (U > 0.15))

    def update(i):
        global A
        U = scipy.signal.convolve2d(A, K, mode='same', boundary='wrap')
        A = np.clip(A + 1 / T * growth(U), 0, 1)
        img.set_array(A)
        return img,

    #figure_asset(K, growth, bar_K=True)
    fig = figure_world(A)
    anim = matplotlib.animation.FuncAnimation(fig, update, frames=20, interval=20).to_jshtml()
    f = "test.gif"
    writergif = matplotlib.animation.PillowWriter(fps=30)
    anim.save(f, writer=writergif)