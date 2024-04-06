with open ("motion_list.txt",'w') as f:
    for i in range(1,3997):
        f.write(f"0{str(i).zfill(4)}.npy\n")