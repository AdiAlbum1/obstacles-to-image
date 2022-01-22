import inference

if __name__ == "__main__":
    # load obstacles
    in_filename, in_filename_2 = "evaluate\\test_scene_(6,-4).json", "evaluate\\test_scene_(4,2).json"

    # calculate ground truth values
    ground_truth_coords, ground_truth_coords_2 = (6, -4), (4, 2)

    _ , res_1 = inference.find_narrow_passageway(in_filename)
    print("GT 1: " + str(ground_truth_coords))
    print("Result 1: " + str(res_1))

    _ , res_2 = inference.find_narrow_passageway(in_filename_2)
    print("GT 2: " + str(ground_truth_coords_2))
    print("Result 2: " + str(res_2))