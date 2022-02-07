import cv2 as cv

import inference

def evaluate(filename):
    img, net_output = inference.find_narrow_passageway(filename)
    pw_start_row, pw_start_col, pw_end_row, pw_end_col = net_output

    pw_start = pw_start_row, pw_start_col
    pw_end = pw_end_row, pw_end_col

    pw_start = pw_start[::-1]
    pw_end = pw_end[::-1]

    # draw box
    color = (255, 0, 0)
    img = cv.rectangle(img, pw_start, pw_end, color)

    cv.imshow("img", img)
    cv.waitKey(0)

if __name__ == "__main__":
    evaluate("evaluate\\test_scene_(4,2).json")
    evaluate("evaluate\\test_scene_(6,-4).json")
    evaluate("evaluate\\test_scene_1.json")
    evaluate("evaluate\\test_scene_2.json")
    evaluate("evaluate\\test_scene_3.json")
    evaluate("evaluate\\test_scene_4.json")