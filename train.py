import cv2 as cv
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import params
from aux_scripts import translate, image_augmenter, passageway_width_calculator

from net import Net
from passageway import Passageway

import mlflow


def load_base_images():
    base_images = {}
    for x_index in range(params.max_num_base_obstacle_maps + 1):
        for i in range(params.max_index_base_obstacle_maps + 1):
            base_obstacle_path = "input_png_obstacles\\" + str(x_index) + "_0\\" + str(i) + ".png"
            base_images[(x_index, i)] = cv.imread(base_obstacle_path, cv.IMREAD_GRAYSCALE)

    return base_images

def load_base_passageway_widths():
    base_passageway_widths = {}
    for x_index in range(params.max_num_base_obstacle_maps + 1):
        for i in range(params.max_index_base_obstacle_maps + 1):
            base_obstacle_path = "input_json_obstacles\\" + str(x_index) + "_0\\" + str(i) + ".json"
            base_passageway_widths[(x_index, i)] = passageway_width_calculator.generate_passageway_width(base_obstacle_path)

    return base_passageway_widths


def generate_batch(batch_size):
    images_batch = []
    labels_batch = []
    for i in range(batch_size):
        # randomly choose a base critical pass
        x_index = random.randint(0, params.max_num_base_obstacle_maps)
        random_base_index = random.randint(0, params.max_index_base_obstacle_maps)

        # read base obstacles
        base_obstacle = base_images[(x_index, random_base_index)].copy()

        # obtain passageway's center: column in pixels
        coordinates = (x_index, 0)
        _, col_pixels = translate.coordinates_to_pixels(coordinates[0], coordinates[1])
        row_pixels = params.im_height / 2
        point_of_interest = (row_pixels, col_pixels)

        # calculate obstacle passageway height and width
        base_passageway_width = base_passageway_widths[(x_index, random_base_index)]
        base_passageway_height = 2 * params.axis_range

        # translate height and width to pixel height and pixel width
        base_passageway_height = (base_passageway_height / (2 * params.axis_range)) * params.im_height
        base_passageway_width = (base_passageway_width / (2 * params.axis_range)) * params.im_width

        # calculate passageway bounding box: start = upper left, end = bottom right
        passageway_start = (row_pixels - (base_passageway_height / 2), col_pixels - (base_passageway_width / 2))
        passageway_end = (row_pixels + (base_passageway_height / 2), col_pixels + (base_passageway_width / 2))
        passageway = Passageway(passageway_start, passageway_end, point_of_interest)

        # randomly translate the base obstacle along the y-axis
        base_obstacle, row_pixels = image_augmenter.translate_along_y_axis(base_obstacle)

        # randomly rotate at {0, 90, 180, 270} degrees
        degrees_lst = [0, 90, 180, 270]
        random.shuffle(degrees_lst)
        rotation_angle = degrees_lst[0]
        base_obstacle = image_augmenter.rotate_image(base_obstacle, rotation_angle)
        row_pixels, col_pixels = translate.translate_pixel_value_for_rotation(row_pixels, col_pixels,
                                                                              params.im_height, params.im_width,
                                                                              rotation_angle)
        passageway_start = translate.translate_pixel_value_for_rotation(passageway_start[0], passageway_start[1],
                                                                        params.im_height, params.im_width,
                                                                        rotation_angle)
        passageway_end = translate.translate_pixel_value_for_rotation(passageway_end[0], passageway_end[1],
                                                                      params.im_height, params.im_width,
                                                                      rotation_angle)

        # start and end change after rotation, fix this:
        passageway_start, passageway_end = translate.fix_box_coordinates(passageway_start, passageway_end, rotation_angle)

        passageway.update_start(passageway_start)
        passageway.update_end(passageway_end)
        passageway.update_point_of_interest((row_pixels, col_pixels))
        passageway.update_is_vertical(rotation_angle)


        # randomly generate additional obstacles
        additional_obstacles, passageway = image_augmenter.randomly_generate_obstacles_avoiding_passageway(row_pixels, col_pixels,
                                                                                               passageway)

        # merge base obstacle with randomly generated obstacles
        all_img = cv.bitwise_or(base_obstacle, additional_obstacles)
        # # draw rectangle
        # passageway_start = passageway.get_start()
        # passageway_end = passageway.get_end()
        # passageway_start = (round(passageway_start[0]), round(passageway_start[1]))
        # passageway_end = (round(passageway_end[0]), round(passageway_end[1]))
        # color = (255, 0, 0)
        # all_img = cv.rectangle(all_img, passageway_start[::-1], passageway_end[::-1], color)
        # cv.imshow("all_img", all_img)
        # cv.waitKey(0)

        # normalize passageway to [0,1] range
        passageway.normalize(params.im_height, params.im_width)

        # normalize image to [0,1] range
        all_img = all_img / 255

        images_batch.append(all_img)
        labels_batch.append((*passageway.get_start(), *passageway.get_end()))

    images_batch = np.array(images_batch)
    labels_batch = np.array(labels_batch)

    images_batch = images_batch.reshape((batch_size, 1, params.im_height, params.im_width))

    return images_batch, labels_batch


if __name__ == "__main__":
    net = Net()
    net.initialize_weights()

    # model_state_dict = torch.load("test_model.pt")
    # net.load_state_dict(model_state_dict)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters())

    base_images = load_base_images()
    base_passageway_widths = load_base_passageway_widths()

    # generate test set
    test_images, test_labels = generate_batch(params.batch_size)
    test_images, test_labels = np.array([test_images]), np.array([test_labels])
    for i in range(params.num_test_set_batches - 1):
        curr_test_images, curr_test_labels = generate_batch(params.batch_size)
        test_images = np.vstack((test_images, np.array([curr_test_images])))
        test_labels = np.vstack((test_labels, np.array([curr_test_labels])))

    test_images, test_labels = torch.from_numpy(test_images), torch.from_numpy(test_labels)

    best_test_loss = 0.02

    with mlflow.start_run():
        train_loss = 0
        # images_batch, labels_batch = generate_batch(params.batch_size)
        # images_batch, labels_batch = torch.from_numpy(images_batch), torch.from_numpy(labels_batch)
        for i in range(1, (params.num_batches_in_epoch + 1)):
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

            # print results
            if i % 200 == 0:
                # current train loss
                curr_train_loss = train_loss / 200

                net.eval()
                # current test loss
                test_loss = 0
                for j in range(params.num_test_set_batches):
                    test_outputs = net(test_images[j].float())
                    curr_test_loss = criterion(test_outputs, test_labels[j].float()).item()
                    test_loss += curr_test_loss
                test_loss = test_loss / params.num_test_set_batches
                net.train()

                # log results
                print(str(i // 200) + "/" + str(params.num_batches_in_epoch // 200) + ":\tTrain loss: " + str(
                    curr_train_loss) + ", Test loss: " + str(test_loss))
                mlflow.log_metrics({"train_loss": curr_train_loss, "test_loss": test_loss})

                # if best test loss improved, save it. Not too often
                if test_loss < best_test_loss:
                    best_test_loss = test_loss
                    torch.save(net.state_dict(), "test_model.pt")
                    print(str(i // 200) + " BEST MODEL - TRAIN LOSS " + str(curr_train_loss) + "\tTEST LOSS " + str(best_test_loss))

                train_loss = 0

        mlflow.log_artifacts("outputs")
        torch.save(net.state_dict(), "test_model.pt")

        print('Finished Training')
