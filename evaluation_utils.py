
from matplotlib import pyplot as plt
import pandas as pd


def capture_evaluation_positions_GUI(img1, img2, coords_img1=[], coords_img2=[]):
    """
    Capture posititions of identical features in a pair of images
    :param img1: the first image of the pair to evaluate the matching on
    :param img2: the second image of the pair to evaluate the matching on
    :param coords_img1: an empty list to store the coordinates in
    :param coords_img2: an empty list to store the coordinates in
    :return: the coordinates of the clicks
    """
    # List for plot elements
    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)

    # toolbar
    tb = plt.get_current_fig_manager().toolbar

    # Show RGB and segmentation mask
    axs[0].imshow(img1)
    axs[0].set_title('Image 1')
    # Show RGB and segmentation mask
    axs[1].imshow(img2)
    axs[1].set_title('Image 2')

    # Drawing functions for redraw
    def draw_training_points():
        # Redraw all
        fig.canvas.draw()

        # Marked positions for Image 1
        if len(coords_img1) > 0:
            df_training_coords_img1 = pd.DataFrame(coords_img1)
            df_training_wheat_img1 = df_training_coords_img1[df_training_coords_img1['set'] == 'Img1']
            df_training_weed_img1 = df_training_coords_img1[df_training_coords_img1['set'] == 'Img2']
            axs[0].scatter('x', 'y', data=df_training_wheat_img1, marker='+', color='white')
            axs[0].scatter('x', 'y', data=df_training_weed_img1, marker='+', color='red')

        # Marked positions for Image 2
        if len(coords_img2) > 0:
            df_training_coords_img2 = pd.DataFrame(coords_img2)
            df_training_wheat_img2 = df_training_coords_img2[df_training_coords_img2['set'] == 'Img1']
            df_training_weed_img2 = df_training_coords_img2[df_training_coords_img2['set'] == 'Img2']
            axs[1].scatter('x', 'y', data=df_training_wheat_img2, marker='+', color='white')
            axs[1].scatter('x', 'y', data=df_training_weed_img2, marker='+', color='red')

    draw_training_points()

    # Event function on click: add or delete training points
    def onclick(event):

        if tb.mode == '':
            # Coordinates of click
            x = event.xdata
            y = event.ydata

            # Identify which image the click occurred on
            clicked_axes = event.inaxes
            if clicked_axes == axs[0]:
                coords_img1.append({'x': x, 'y': y, 'set': 'Img1'})
            elif clicked_axes == axs[1]:
                coords_img2.append({'x': x, 'y': y, 'set': 'Img2'})

            print('last entry:', coords_img1[-1] if clicked_axes == axs[0] else coords_img2[-1])

            # Remove all points from graph
            for plot_id in range(len(axs)):
                del axs[plot_id].collections[:]

            # Redraw graph
            draw_training_points()

    # Handle mouse click and keyboard events
    fig.canvas.mpl_connect('button_press_event', onclick)

    # Start GUI
    plt.interactive(True)
    plt.show(block=True)

    plt.interactive(False)

    # Return captured coordinates for both images
    return coords_img1, coords_img2
