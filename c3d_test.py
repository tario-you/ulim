import c3d

with open('dataset/2017-06-22/00001_raw.c3d', 'rb') as handle:
    reader = c3d.Reader(handle)
    frames = 0
    for i, k in enumerate(reader.read_frames()): 
        print(k)
        frames += 1

print(frames)