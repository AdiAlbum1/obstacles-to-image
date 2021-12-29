import cv2 as cv
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import params
from aux_scripts import translate, image_augmenter

import mlflow

def generate_batch(batch_size):
    images_batch = []
    labels_batch = []
    for i in range(batch_size):
        # randomly choose a base critical pass
        x_index = random.randint(0, params.max_num_base_obstacle_maps)
        random_base_index = random.randint(0, params.max_index_base_obstacle_maps)

        # read base obstacle
        base_obstacle_path = "input_png_obstacles\\"+str(x_index)+"_0\\"+str(random_base_index)+".png"
        base_obstacle = cv.imread(base_obstacle_path, cv.IMREAD_GRAYSCALE)

        # randomly translate the base obstacle along the y-axis
        base_obstacle, pixels_row = image_augmenter.translate_along_y_axis(base_obstacle)

        # obtain pixels_col
        coordinates = (x_index, 0)
        _, pixels_col = translate.coordinates_to_pixels(coordinates[0], coordinates[1])

        # randomly rotate at {0, 90, 180, 270} degrees
        degrees_lst = [0, 90, 180, 270]
        random.shuffle(degrees_lst)
        rotation_angle = degrees_lst[0]
        base_obstacle = image_augmenter.rotate_image(base_obstacle, rotation_angle)
        pixels_row, pixels_col = translate.translate_pixel_value_for_rotation(pixels_row, pixels_col, params.im_height, params.im_width, rotation_angle)

        # randomly generate additional obstacles
        additional_obstacles = image_augmenter.randomly_generate_obstacles_avoiding_passageway(pixels_row, pixels_col)

        # merge base obstacle with randomly generated obstacles
        all_img = cv.bitwise_or(base_obstacle, additional_obstacles)

        # normalize image to [0,1] range
        all_img = all_img / 255

        # normalize row and col to [0,1] range
        normalized_pixel_row = pixels_row / params.im_height
        normalized_pixel_col = pixels_col / params.im_width

        images_batch.append(all_img)
        labels_batch.append((normalized_pixel_row, normalized_pixel_col))

    images_batch = np.array(images_batch)
    labels_batch = np.array(labels_batch)

    images_batch = images_batch.reshape((batch_size, 1, params.im_height, params.im_width))

    return images_batch, labels_batch

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 16, 5)
        self.conv3 = nn.Conv2d(16, 8, 5)
        self.conv4 = nn.Conv2d(8, 4, 5)
        self.fc1 = nn.Linear(4 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 60)
        self.fc3 = nn.Linear(60, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    net = Net()

    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    test_images, test_labels = generate_batch(params.test_set_size)
    test_images, test_labels = torch.from_numpy(test_images), torch.from_numpy(test_labels)

    with mlflow.start_run():
        for epoch in range(params.num_epochs):

            running_loss = 0.0
            train_loss = 0
            for i in range(params.num_batches_in_epoch):
                # generate batch
                images_batch, labels_batch = generate_batch(params.batch_size)
                images_batch, labels_batch = torch.from_numpy(images_batch), torch.from_numpy(labels_batch)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(images_batch.float())
                loss = criterion(outputs, labels_batch.float())
                train_loss += loss.item()
                loss.backward()
                optimizer.step()


            train_loss = train_loss / params.num_batches_in_epoch
            # calculate test loss per epoch
            test_outputs = net(test_images.float())
            test_loss = criterion(test_outputs, test_labels.float()).item()
            print("TEST LOSS: " + str(test_loss))

            mlflow.log_metrics({"train_loss" : train_loss, "test_loss" : test_loss})
        mlflow.log_artifacts("outputs")
        torch.save(net, "test_model.pt")

        print('Finished Training')
