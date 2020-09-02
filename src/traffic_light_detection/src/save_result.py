from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
import numpy as np

def save_images(img, detections, save_path, classes, cls_thresh, img_size=512):
    color = ['orangered', 'orangered', 'orangered', 'orangered', 'orangered', 'g', 'g', 'g', 'g', 'g',
             'mediumslateblue']
    print('\nSaving images:')
    # Iterate through images and save plot of detections

    plt.figure(figsize=(6.8, 4), dpi=100)
    fig, ax = plt.subplots()
    fig.set_size_inches(6.8, 4)
    ax.imshow(img)

    # The amount of padding that was added
    pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
    # Image height and width after padding is removed
    unpad_h = img_size - pad_y
    unpad_w = img_size - pad_x

    # Draw bounding boxes and labels of detections
    if detections is not None:
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            if y2 > img_size / 2:
                continue
            if cls_conf > cls_thresh:
                print('\t+ Label: %s, Cls_Conf: %.5f' % (classes[int(cls_pred)], cls_conf.item()))

                # Rescale coordinates to original dimensions
                box_h = ((y2 - y1) / unpad_h) * img.shape[0]
                box_w = ((x2 - x1) / unpad_w) * img.shape[1]
                y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
                x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]

                # Create a Rectangle patch
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2,
                                         edgecolor=color[int(cls_pred)],
                                         facecolor='none')

                # Add the bbox to the plot
                ax.add_patch(bbox)
                # Add label
                plt.text(x1, y1, s=classes[int(cls_pred)] + ' %.2f' % (float(cls_conf.cpu())), color='white',
                         verticalalignment='bottom',
                         bbox={'color': color[int(cls_pred)], 'pad': 0}, fontsize='xx-small')

    # Save generated image with detections
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    # plt.savefig('output/%d.png' % (img_i), bbox_inches='tight', pad_inches=0.0)
    plt.savefig(save_path, pad_inches=0.0)
    plt.close()