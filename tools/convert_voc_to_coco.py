import os
import argparse
import json
import xml.etree.ElementTree as ET
from typing import Dict, List
from tqdm import tqdm
import re

LABELS = {'구진': 1,
          '농포': 2,
          '결절': 3,
          '낭포': 4,
          '결절/낭포': 5,
          '켈로이드': 6,
          '화이트헤드': 7,
          '블랙헤드': 8,
          '모낭염': 9,
          '여드름자국': 10,
          '여드름흉터': 11,
          '표피낭종': 12,
          }


def make_dir(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


def get_annpaths(ann_dir_path: str = None) -> List[str]:
    print()
    ann_files = os.listdir(ann_dir_path)
    ann_paths = [os.path.join(ann_dir_path, f) for f in ann_files]
    return ann_paths


def get_image_info(annotation_root, extract_num_from_imgid=True):
    path = annotation_root.findtext('path')
    if path is None:
        filename = annotation_root.findtext('filename')
    else:
        filename = os.path.basename(path)
    img_name = os.path.basename(filename)
    img_id = os.path.splitext(img_name)[0]
    if extract_num_from_imgid and isinstance(img_id, str):
        img_id = int(re.findall(r'\d+', img_id)[-1])

    size = annotation_root.find('size')
    width = int(size.findtext('width'))
    height = int(size.findtext('height'))

    image_info = {
        'file_name': filename,
        'height': height,
        'width': width,
        'id': img_id
    }
    return image_info


def get_coco_annotation_from_obj(obj, label2id):
    label = obj.findtext('name')
    assert label in label2id, f"Error: {label} is not in label2id !"
    category_id = label2id[label]
    bndbox = obj.find('bndbox')
    xmin = int(bndbox.findtext('xmin')) - 1
    ymin = int(bndbox.findtext('ymin')) - 1
    xmax = int(bndbox.findtext('xmax'))
    ymax = int(bndbox.findtext('ymax'))
    assert xmax > xmin and ymax > ymin, f"Box size error !: (xmin, ymin, xmax, ymax): {xmin, ymin, xmax, ymax}"
    o_width = xmax - xmin
    o_height = ymax - ymin
    ann = {
        'bbox': [xmin, ymin, o_width, o_height],
        'area': o_width * o_height,
        'category_id': category_id,
        'ignore': 0,
        'iscrowd': 0,
    }
    return ann


def convert_xmls_to_cocojson(annotation_paths: List[str],
                             label2id: Dict[str, int],
                             output_jsonpath: str,
                             extract_num_from_imgid: bool = True):
    output_json_dict = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    bnd_id = 1  # START_BOUNDING_BOX_ID, TODO input as args ?
    print('Start converting !')
    for a_path in tqdm(annotation_paths):
        # Read annotation xml
        ann_tree = ET.parse(a_path)
        ann_root = ann_tree.getroot()

        img_info = get_image_info(annotation_root=ann_root,
                                  extract_num_from_imgid=extract_num_from_imgid)
        img_id = img_info['id']
        output_json_dict['images'].append(img_info)

        for obj in ann_root.findall('object'):
            ann = get_coco_annotation_from_obj(obj=obj, label2id=label2id)
            ann.update({'image_id': img_id, 'id': bnd_id})
            output_json_dict['annotations'].append(ann)
            bnd_id = bnd_id + 1

    for label, label_id in label2id.items():
        category_info = {'supercategory': 'none', 'id': label_id, 'name': label}
        output_json_dict['categories'].append(category_info)

    make_dir(output_jsonpath)
    with open(os.path.join(output_jsonpath, 'output.json'), 'w', encoding='utf-8') as f:
        output_json = json.dumps(output_json_dict)
        f.write(output_json)


def main():
    parser = argparse.ArgumentParser(
        description='This script support converting pascal voc format xmls to coco format json')
    parser.add_argument('--voc_dir', type=str, default=None,
                        help='path to pascal voc annotation files directory.')
    parser.add_argument('--coco_dir', type=str, default='output.json', help='path to coco annotation files directory.')
    args = parser.parse_args()
    convert_xmls_to_cocojson(
        annotation_paths=get_annpaths(ann_dir_path=args.voc_dir),
        label2id=LABELS,
        output_jsonpath=args.coco_dir,
        extract_num_from_imgid=True
    )


if __name__ == '__main__':
    main()