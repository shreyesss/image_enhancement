

from multiprocessing import Pool
parts = []
paths = *zip(left_paths,right_paths)
chunks =10
n = len(paths)/chunks
for i in range(0,len(paths),n):
    dict_ = {}
    dict_["proc"] = i
    dict_["input"] = paths[i:i+n]
    parts.append(dict_)

def process_images(chunk):
    output_path = "./fixed/{}/".format(chunk["proc"])
    for i,img in enumerate(dict_[input_]):
        left = img[0]
        right = img[1]
        out_left,out_right = convert(left,right)

        #save left and right

pool = Pool(processes=chunks)
pool.map(process_images,parts )