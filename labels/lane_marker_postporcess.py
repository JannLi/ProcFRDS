import cv2
import os
import numpy as np
from shapely import Polygon

def get_polygon(label, sample):
    contours, hierarchy = cv2.findContours(label, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for i in range(len(contours)):
        contour = contours[i].copy()
        approx = cv2.approxPolyDP(contour, sample, True)
        polygon = []
        for p in approx:
            polygon.append(p[0].tolist())
        polygons.append(polygon)
    return polygons

def filter_polygon_area(polygons, threshold):
    selected = []
    for polygon in polygons:
        if len(polygon) <= 3:
            continue
        poly = Polygon(polygon).buffer(0.01)
        if poly.area > threshold:
            selected.append(polygon)
    return selected

def get_instance_mask(seg_mask, sample):
    seg_mask[seg_mask!=0] = 255
    seg_mask = cv2.cvtColor(seg_mask, cv2.COLOR_BGR2GRAY)
    polygons = get_polygon(seg_mask, sample)
    instance_mask = np.zeros(seg_mask.shape, dtype=np.uint8)
    instance_count = 1
    for poly in polygons:
        poly = np.array(np.array(poly).astype(np.float32), dtype=np.int32)
        cv2.fillPoly(instance_mask, [poly], (instance_count, instance_count, instance_count))
        instance_count += 1
    return instance_mask

def mask_process(img, color_dict):
    mask_processed = np.zeros(img.shape, dtype=np.uint8)
    thred_near = 70
    thred_far = 120
    for key in color_dict:
        color, true_color = color_dict[key]
        if key == 'red':
            thred1, thred2, thred3 = thred_near, thred_near, thred_far 
        elif key == 'blue':
            thred1, thred2, thred3 = thred_far, thred_near, thred_near
        # elif key == 'green':
        #     thred1, thred2, thred3 = thred_near, thred_far, thred_far
        else:
            thred1, thred2, thred3 = thred_near, thred_near, thred_near
        lowbgr = (max(color[0]-thred1, 0), max(color[1]-thred2, 0), max(color[2]-thred3, 0))
        highbgr = (min(color[0]+thred1, 255), min(color[1]+thred2, 255), min(color[2]+thred3, 255))
        mask = cv2.inRange(img, lowerb=lowbgr, upperb=highbgr)
        polygons = get_polygon(mask, 0.5)
        filtered_polygons = filter_polygon_area(polygons, 100)
        # filtered_polygons = polygons
        for poly in filtered_polygons:
            poly = np.array(np.array(poly).astype(np.float32), dtype=np.int32)
            cv2.fillPoly(mask_processed, [poly], true_color)
    return mask_processed

def main(args):
    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)
    solid_line = (1, 1, 1)
    dashed_line = (2, 2, 2)
    line_type = args.line_type
    if line_type[0] == 's':
        line1 = solid_line
    else:
        line1 = dashed_line
    if line_type[1] == 's':
        line2 = solid_line
    else:
        line2 = dashed_line
    color_dict = dict()
    color_dict['red'] = [(33, 58, 219), line2]
    color_dict['green'] = [(82, 197, 111), line1]
    color_dict['green2'] = [(40, 150, 60), line1]
    # color_dict['yellow'] = [(109, 194, 200), (0, 255, 255)]
    color_dict['white'] = [(197, 197, 197), (3, 3, 3)]
    color_dict['blue'] = [(209, 86, 20), (1, 1, 1)]
    names = os.listdir(args.mask_path)
    for i in range(len(names)):
        name = names[i]
        if not name.startswith('fisheye'):
            continue
        print(name+'---------------'+str(i)+'/'+str(len(names)))
        mask = cv2.imread(os.path.join(args.mask_path, name))
        processed_mask = mask_process(mask, color_dict)
        cv2.imwrite(os.path.join(args.out_path, name.replace('.png', '_seg.png')), processed_mask)
        instance_mask = get_instance_mask(processed_mask, 0.5)
        cv2.imwrite(os.path.join(args.out_path, name.replace('.png', '_instance.png')), instance_mask)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='lane marker postprocess')
    parser.add_argument('--mask_path', help='mask path')
    parser.add_argument('--line_type', help='line_type')
    parser.add_argument('--out_path', help='output path')
    args = parser.parse_args()

    main(args)
