import numpy as np
import os
import torch
import collections
from torchvision import transforms

# from datasets import ImageFolder, ListDataset
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches

class my_transform(object):

    def __init__(self, target_transform_5crops = False, target_transform_original_image = False,
                 crops_size=(800, 1280), image_size=(1200, 1920), input_size=512, fill=0, max_boxes_per_img=5):
        assert isinstance(target_transform_5crops, bool) and isinstance(target_transform_original_image, bool)\
               and (target_transform_original_image == True and target_transform_5crops == True) \
               and isinstance(crops_size, int) or (isinstance(crops_size, collections.Iterable) and len(crops_size) == 2)
        self.target_transform_5crops = target_transform_5crops
        self.target_transform_original_image = target_transform_original_image
        self.input_size = input_size
        self.fill = fill
        self.max_objects = max_boxes_per_img
        self.image_size = image_size
        self.h, self.w = self.image_size
        self.crops_size = crops_size
        self.crop_h, self.crop_w = self.crops_size
        dim_diff = np.abs(self.crop_h - self.crop_w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        self.crop_pad = (0, pad1, 0, pad2) if self.crop_h <= self.crop_w else (pad1, 0, pad2, 0)

        dim_diff = np.abs(self.h - self.w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        self.pad = (0, pad1, 0, pad2) if self.h <= self.w else (pad1, 0, pad2, 0)

    # def __call__(self, labels):
    #     if self.target_transform_5crops:
    #         return self.crops5(labels)
    #     elif self.target_transform_original_image:
    #         return self.org(labels)

    def target_transform_crops5(self, label_path):
        labels = None
        if os.path.exists(label_path):
            labels = np.loadtxt(label_path).reshape(-1, 5)
            if labels is not None:
                labels[range(len(labels))[:self.max_objects]] = labels[:self.max_objects]
            # Extract coordinates for unpadded + unscaled image
            x1 = self.w * (labels[:, 1] - labels[:, 3] / 2)
            y1 = self.h * (labels[:, 2] - labels[:, 4] / 2)
            x2 = self.w * (labels[:, 1] + labels[:, 3] / 2)
            y2 = self.h * (labels[:, 2] + labels[:, 4] / 2)
            tl = (0, 0, self.crop_w, self.crop_h)
            tr = (self.w - self.crop_w, 0, self.w, self.crop_h)
            bl = (0, self.h - self.crop_h, self.crop_w, self.h)
            br = (self.w - self.crop_w, self.h - self.crop_h, self.w, self.h)
            center = ((self.w-self.crop_w)//2, (self.h-self.crop_h)//2, (self.w+self.crop_w)//2, (self.h+self.crop_h)//2)
            crops_edge = (tl, tr, bl, br, center)
            # Adjust for added padding
            padded_w = float(self.crop_w + 2 * self.crop_pad[0])
            padded_h = float(self.crop_h + 2 * self.crop_pad[1])
            crops_labels = np.zeros((5, self.max_objects, 5), float)
            for i in range(5):
                for j in range(min(labels.shape[0], self.max_objects)):
                    if x1[j] > crops_edge[i][0] and y1[j] > crops_edge[i][1] and x2[j] < crops_edge[i][2] and y2[j] < crops_edge[i][3]:
                        crops_x1 = x1[j] - crops_edge[i][0] + self.crop_pad[0]
                        crops_y1 = y1[j] - crops_edge[i][1] + self.crop_pad[1]
                        crops_x2 = x2[j] - crops_edge[i][0] + self.crop_pad[0]
                        crops_y2 = y2[j] - crops_edge[i][1] + self.crop_pad[1]
                        # Calculate ratios from coordinates
                        crops_labels[i, j, 0] = labels[j, 0]
                        crops_labels[i, j, 1] = ((crops_x1 + crops_x2) / 2.) / padded_w
                        crops_labels[i, j, 2] = ((crops_y1 + crops_y2) / 2.) / padded_h
                        crops_labels[i, j, 3] = labels[j, 3] * self.w / padded_w
                        crops_labels[i, j, 4] = labels[j, 4] * self.h / padded_h
                        # print(crops_labels)


        fill_labels = torch.from_numpy(crops_labels)
        return fill_labels

    def target_transform_org(self, label_path):
        labels = None

        if os.path.getsize(label_path) == 0:
            filled_labels = np.zeros((self.max_objects, 5))
            return torch.from_numpy(filled_labels)

        if os.path.exists(label_path):
            labels = np.loadtxt(label_path).reshape(-1, 5)
            # Extract coordinates for unpadded + unscaled image
            x1 = self.w * (labels[:, 1] - labels[:, 3]/2)
            y1 = self.h * (labels[:, 2] - labels[:, 4]/2)
            x2 = self.w * (labels[:, 1] + labels[:, 3]/2)
            y2 = self.h * (labels[:, 2] + labels[:, 4]/2)
            # Adjust for added padding
            x1 += self.pad[0]
            y1 += self.pad[1]
            x2 += self.pad[0]
            y2 += self.pad[1]
            padded_w = self.w + 2*self.pad[0]
            padded_h = self.h + 2*self.pad[1]
            # Calculate ratios from coordinates
            labels[:, 1] = ((x1 + x2) / 2) / padded_w
            labels[:, 2] = ((y1 + y2) / 2) / padded_h
            labels[:, 3] *= self.w / padded_w
            labels[:, 4] *= self.h / padded_h
        # Fill matrix
        filled_labels = np.zeros((self.max_objects, 5))
        if labels is not None:
            filled_labels[range(len(labels))[:self.max_objects]] = labels[:self.max_objects]
        filled_labels = torch.from_numpy(filled_labels)
        return filled_labels

    def image_transform_5crops(self):

        return transforms.Compose([
            transforms.FiveCrop(self.crops_size),
            transforms.Lambda(lambda crops: torch.stack(
                [transforms.ToTensor()(transforms.Resize((self.input_size,self.input_size))
                                       (transforms.Pad(self.crop_pad, fill=self.fill)(crop))) for crop in crops]))
        ])

    def image_transforms_original_image(self):
        return transforms.Compose([
            transforms.Pad(self.pad, fill=self.fill),
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor()
        ])

    def __repr__(self):
        return self.__class__.__name__ + '()'


# if __name__ == '__main__':
#     input_size = 512
#     batch_size = 1
#     train_path = '/home/iaircv/DATA/traffic_light_data/TLdataset/train.txt'
#     # transform = my_transform(target_transform_original_image=True)
#     # # Get dataloader
#     # dataloader = torch.utils.data.DataLoader(
#     #     ListDataset(train_path, input_size, transform=transform.image_transforms_original_image(),
#     # target_transform=transform),
#     #     batch_size=batch_size, shuffle=False, num_workers=8)
#
#     # transform = my_transform(target_transform_5crops=True)
#     # dataloader = torch.utils.data.DataLoader(
#     #     ListDataset(train_path, input_size, transform=transform.image_transform_5crops(),
#     #                 target_transform=transform),
#     #     batch_size=batch_size, shuffle=False, num_workers=8)
#     transform = my_transform()
#     dataloader = torch.utils.data.DataLoader(
#         ListDataset(train_path, input_size,
#                     transform=transform.image_transform_5crops(),
#                     transform2=transform.image_transforms_original_image(),
#                     target_transform=transform.target_transform_crops5,
#                     target_transform2=transform.target_transform_org),
#         batch_size=batch_size, shuffle=False, num_workers=8)
#
#     for batch_i, (_, imgs, targets) in enumerate(dataloader):
#         print(imgs.size())
#         for i in range(6):
#             image = imgs[0, i, :, :, :]
#             fig, ax = plt.subplots()
#             ax.imshow(np.transpose(image, (1, 2, 0)))
#             # plt.savefig("/home/iaircv/PycharmProjects/traffic_lights/code0806/output/%d.jpg" % i)
#             label = targets[0, i, 0, :]*input_size
#             x1 = label[1] - label[3]/2
#             y1 = label[2] - label[4]/2
#             x2 = label[1] + label[3]/2
#             y2 = label[2] + label[4]/2
#             bbox = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1,
#                                      edgecolor='r',
#                                      facecolor='none')
#             ax.add_patch(bbox)
#             plt.savefig("/home/iaircv/PycharmProjects/traffic_lights/code0806/output/%d.jpg" % i)
#             print(label)
#         print(targets.size())
