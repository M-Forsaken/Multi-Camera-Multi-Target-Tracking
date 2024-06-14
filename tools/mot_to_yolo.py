import shutil
import argparse
import os
from pathlib import Path

import yaml

cwd = os.getcwd()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name", default="MOT17Det", help="Enter dataset name."
    )
    args = parser.parse_args()
    name = args.name
    path = cwd + "/datasets/" + name

    seqnames = os.listdir(path)
    for seq in seqnames:
        count = 0
        img_path = path + "/" + seq + "/img1/"
        img_list = os.listdir(img_path)
        txt_path = path + "/" + seq + "/gt/gt.txt"
        txt = open(txt_path, 'r+')
        out_path = path + "/" + seq + \
            "/labels/"
        object_list = [[] for _ in range(len(img_list))]
        for line in txt:
            values = []
            value = ""
            for c in line:
                if c != "," and c != "\n":
                    value += c
                else:
                    values.append(float(value))
                    value = ""
            frame_count, id, top, left, width, height, _, _, _ = values
            object_list[int(frame_count) - 1].append(values)
        while count < len(img_list):
            out_file = out_path + \
                os.path.splitext(os.path.basename(
                    img_list[count]))[0] + ".txt"
            Path(out_file).parent.mkdir(parents=True, exist_ok=True)
            out_txt = open(out_file, 'w')
            for object in object_list[count]:
                frame_count, id, top, left, width, height, _, _, _ = object
                out_txt.write(
                    f'0 {top} {left} {width} {height}\n')
            count += 1
            out_txt.close()
        txt.close()
        new_path = path + "/" + seq 
        os.makedirs(new_path + "/train/images", exist_ok=True)
        os.makedirs(new_path + "/train/labels", exist_ok=True)
        os.makedirs(new_path + "/val/images", exist_ok=True)
        os.makedirs(new_path + "/val/labels", exist_ok=True)
        for idx, img in enumerate(img_list):
            if idx < round(len(img_list)*8/10):
                shutil.move(img_path + img, new_path + "/train/images/" + img)
                shutil.move(out_path + os.path.splitext(os.path.basename(img))[
                            0] + ".txt", new_path + "/train/labels/" + os.path.splitext(os.path.basename(img))[0] + ".txt")
            else:
                shutil.move(img_path + img, new_path + "/val/images/" + img)
                shutil.move(out_path + os.path.splitext(os.path.basename(img))[
                            0] + ".txt", new_path + "/val/labels/" + os.path.splitext(os.path.basename(img))[0] + ".txt")
        os.removedirs(img_path)
        os.removedirs(out_path)
        os.remove(txt_path)
        os.removedirs(path + "/" + seq + "/gt")
        data = dict(
            path = new_path,
            train="train/images",
            val="val/images",
            nc= 1,
            names= ['person'],
        )
        yaml_path = path + "/" + seq + "/config.yaml"
        Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)
        with open(yaml_path, 'w') as outfile:
            yaml.dump(data, outfile)

