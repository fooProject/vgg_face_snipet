# _*_ coding: utf-8 _*_
"""
@author 张垚

usage:
python crop_face.py <dlib_landmark_model_path> <source_image_path> <dst_image_path>
example:
python crop_face.py ./shape_predictor_68_face_landmarks.dat ./srd_images ./dst_images
"""

import cv2
import sys
import os
import dlib
import time
from skimage import io

def getBoundingBox(left_eye, right_eye, nose_tip, left_mouth, right_mouth):
    left = left_eye[0]
    if left > left_mouth[0]:
        left = left_mouth[0]
    top = left_eye[1]
    if top > right_eye[1]:
        top = right_eye[1]
        
    right = right_eye[0]
    if right < right_mouth[0]:
        right = right_eye[0]
    bot = left_mouth[1]
    if bot < right_mouth[1]:
        bot = right_mouth[1]
    
    return (int(left), int(top), int(right), int(bot))


def scaleFaceBox(box, size, factor = 3.4):
    left = box[0]
    top = box[1]
    right = box[2]
    bottom = box[3]
    w = size[0]
    h = size[1]
    
    bw = right - left
    bh = bottom - top
    
    longEdge = bw
    if longEdge < bh:
        longEdge = bh
    
    longEdge = longEdge * factor
    
    left -= 0.5 * (longEdge - bw)
    top -= 0.58 * (longEdge - bh)
    right = left + longEdge
    bottom = top + longEdge
    if left < 0:
        left = 0
    if top < 0:
        top = 0
    if right > w:
        right = w
    if bottom > h:
        bottom = h
        
    return (int(left), int(top), int(right), int(bottom))

def main(argv):
    dlib_face_landmark_model_filename = argv[0]
    source_image_dir = argv[1]
    dst_image_dir = argv[2]

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(dlib_face_landmark_model_filename)
 
    count = 0
    for root, subDir, files in os.walk(source_image_dir):
        for file in files:
            f = os.path.join(root, file)
            if not f.endswith('.jpg'):
                continue
            txtPath = f.replace('.jpg', '.txt')
            if os.path.isfile(txtPath):
                continue
                
            #print("Processing file: {}".format(f))
            image = cv2.imread(f)
            
            size = image.shape
            
            img = io.imread(f)

            # Ask the detector to find the bounding boxes of each face. The 1 in the
            # second argument indicates that we should upsample the image 1 time. This
            # will make everything bigger and allow us to detect more faces.
            start_time = time.time()
            dets = detector(img)
            elapsed_time = time.time() - start_time
            
    
            index = 0
            dw = -1
            for k, d in enumerate(dets):
                if dw < (d.right() - d.left()):
                    dw = d.right() - d.left()
                    index = k
            print("Image {} detected {} faces; elapsed {} ms".format(file, len(dets), 1000*elapsed_time))
    
            for k, d in enumerate(dets):
                
                if k != index:
                    continue
                print("max face {}:".format((index+1)))
                
                shape = predictor(img, d)
    
                acx = 0
                acy = 0
                for n in range(36, 42):
                    acx += shape.part(n).x
                    acy += shape.part(n).y
                lx = float(acx) / 6
                ly = float(acy) / 6
                #fin.write("%f %f\n" % (lx, ly))
                #cv2.circle(image, (int(lx), int(ly)), 2, (255, 255, 0))
    
                acx = 0
                acy = 0
                for n in range(42, 48):
                    acx += shape.part(n).x
                    acy += shape.part(n).y
                rx = float(acx) / 6
                ry = float(acy) /6
                #fin.write("%f %f\n" % (rx, ry))
                #cv2.circle(image, (int(rx), int(ry)), 2, (255, 255, 0))
    
                nosex = shape.part(30).x
                nosey = shape.part(30).y
                #fin.write("%f %f\n" % (nosex, nosey))
                #cv2.circle(image, (int(nosex), int(nosey)), 2, (255, 255, 0))
    
    
                mouth_lx = shape.part(48).x
                mouth_ly = shape.part(48).y
                #fin.write("%f %f\n" % (mouth_lx, mouth_ly))
                #cv2.circle(image, (int(mouth_lx), int(mouth_ly)), 2, (255, 255, 0))
    
    
                mouth_rx = shape.part(54).x
                mouth_ry = shape.part(54).y
                #fin.write("%f %f\n" % (mouth_rx, mouth_ry))
                #cv2.circle(image, (int(mouth_rx), int(mouth_ry)), 2, (255, 255, 0))
    
                box = getBoundingBox((lx, ly), (rx, ry), (nosex, nosey), (mouth_lx, mouth_ly), (mouth_rx, mouth_ry))
                faceBox = scaleFaceBox(box, (size[1], size[0]))
                #cv2.rectangle(image, (faceBox[0], faceBox[1]), (faceBox[2], faceBox[3]), (255, 255, 0), 2)
                
                face = image[faceBox[1]:faceBox[3], faceBox[0]:faceBox[2], :]
                normFace = cv2.resize(face, (224, 224))
                
                cv2.imwrite(os.path.join(dst_image_dir, file), normFace)
                count += 1
    print ("Totally processed: {} images!".format(count))

if __name__=='__main__':
    main(sys.argv[1:])
