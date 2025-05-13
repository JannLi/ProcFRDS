import cv2
import numpy as np

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

def main(index, offset):
    img = cv2.imread('./loc2/15/test/avm/avm_2023-11-18-02-02-00_23315000000.jpg')
    img[600:690, 590:610, :] = 0
    # ori = cv2.imread('C://Users//yncmbi//Downloads//avm_2023-11-18-02-02-00_2330000000_2.jpg')
    # bgrs = [(255, 0, 0), (0, 255, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
    ori = np.zeros(img.shape)
    bgrs = [(0, 0, 255)]
    thred = 220
    pts = []
    for i in range(len(bgrs)):
        pt = []
        bgr = bgrs[i]
        lower = (max(0, bgr[0]-thred), max(0, bgr[1]-thred), max(0, bgr[2]-thred))
        upper = (min(255, bgr[0]+thred), min(255, bgr[1]+thred), min(255, bgr[2]+thred))
        mask2 = cv2.inRange(img, lowerb=lower, upperb=upper)
        polygons = get_polygon(mask2, 1)
        for poly in polygons:
            temp = np.average(np.array(poly), axis=0).tolist()
            temp = [int(item) for item in temp]
            cv2.circle(ori, temp, 2, (255, 255, 255), 3)
            pt.append(temp)
        pts.append(pt)
        cv2.imwrite('./loc2/15/test/test_'+str(i)+'.png', mask2)
    cv2.imwrite('./loc2/15/test/aaa.png', ori)

    true_center = [544+offset[0], 679-offset[1]]
    true_locs = []
    avm_loc1 = []
    for i in range(index-1):
        for j in range(index):
            loc = [true_center[0]+int(50*(-10+20/(index-1)*i)), true_center[1]+int(50*(-10+20/(index-1)*j))]
            if i in [6,7] and j in [6,7,8,9]:
                avm_loc1.append(loc)
            true_locs.append(loc)
    np.save('./loc2/15/test/true_loc_'+str(index)+'_'+str(offset[0])+'_'+str(offset[1])+'.npy', np.array(true_locs))
    print(true_locs)
    print(len(true_locs))

    sort_pts = []
    pts = pts[0]+avm_loc1
    # pts.append([1042, 875])
    # pts.remove([907, 914])
    pts.sort(key=lambda coord: (coord[0], coord[1]))
    for i in range(index):
        pts_temp = pts[index*i:index*i+index]
        pts_temp.sort(key=lambda coord: (coord[1], coord[0]))
        sort_pts += pts_temp
    print(sort_pts)
    print(len(sort_pts))
    print((np.array(true_locs)-np.array(sort_pts)).tolist())

    np.save('./loc2/15/test/avm_loc_'+str(index)+'_'+str(offset[0])+'_'+str(offset[1])+'.npy', np.array(sort_pts))

if __name__ == '__main__':
    # dist = [0, 20, 40, 60]
    # for i in dist:
    #     for j in dist:
    #         offset = [i, -j]
    offset = [60, 60]
    main(16, offset)

