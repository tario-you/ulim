import numpy as np

# Replace 'your_file_path.npy' with the path to your actual .npy file
for i in range(1,2):
    file_path = f'dataset/KIT-ML/new_joints/0000{i}.npy'

    # Load the .npy file
    data = np.load(file_path)
    print(data.shape, file_path)

    # if "vec" not in file_path:
    #     frames, _, _ = data.shape
    # else:
    #     frames, _ = data.shape

    # Print the contents of the loaded .npy file
    # print(data)

#KIT = 251