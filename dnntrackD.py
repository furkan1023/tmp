import time
import cv2
from multiprocessing import Process, Manager, Value, Lock
from tensorflow.keras import Input, Model
from darknet import darknet_base
from predict import predict, predict_with_yolo_head
from multiprocessing import Process
from ctypes import c_char_p
import numpy as np
import argparse
import time
import cv2
import os
import sys
import dlib

c=0
applyMask = False
video2 = True
chkMaskIntersection = True




def kutuKarsilastir(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    kosul = 0
    if xA > xB:
        kosul = 1
    if yB < yA:
        kosul = 1

    if kosul == 0:
        interArea = (xB - xA) * (yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        if (boxBArea > boxAArea):
            iou = interArea / boxAArea
        else:
            iou = interArea / boxBArea

        return iou
    else:
        return 0

def detect(rtcars, fr, detStarter, modelLoaded, lock,fp):
    mask = cv2.imread(r"mask\mask.bmp", 0)
    labelsPath = r"data\coco.names"
    weightsPath = (r"cfg\yolov3-tinycoco.weights")
    configPath = (r"cfg\yolov3-tinycoco.cfg")

    LABELS = open(labelsPath).read().strip().split("\n")
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                               dtype="uint8")

    (W, H) = (None, None)
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    modelLoaded.value = True
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    while True:
        if (detStarter.value == True):
            mousePoints = []
            st = time.time()
            frame = fr.value

            if W is None or H is None:
                (H, W) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                         swapRB=True, crop=False)
            net.setInput(blob)
            layerOutputs = net.forward(ln)
            boxes = []
            confidences = []
            classIDs = []
            clases=[]
            for output in layerOutputs:
                for detection in output:
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

            idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.3)

            print("len(idxs): ", len(idxs))
            realTimeCars = []
            aboveTh = 0
            if len(idxs) > 0:
                for i in idxs.flatten():
                    (x, y, w, h) = (boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3])

                    roi = mask[y:y + h, x:x + w]
                    if not chkMaskIntersection or (cv2.countNonZero(roi) / (w * h) > .2) and y+h<350:
                            realTimeCars.append([x, y, w,  h])
                            clases.append(classIDs[i])

            rtcars.value = [realTimeCars, clases]
            detStarter.value = False

            lock.value = True
            fp.value=1/(time.time()-st)
            print(time.time()-st)





def tracking(rtcars, fr, detStarter, modelLoaded, lock,trackers,lock2,seritler,fp):
    xFinish=1280
    xStart=274
    yStart=350
    yFinish=350
    for i in range(len(sys.argv[1:])):
        if i == 0:
            xFinish = sys.argv[1] - 250
    labelsPath = r"yolo\coco.names"

    outputPath=r'output\output.avi'
    cap = cv2.VideoCapture(r"3.mp4")
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    LABELS = open(labelsPath).read().strip().split("\n")
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                               dtype="uint8")

    OPENCV_OBJECT_TRACKERS = {
        "csrt": cv2.TrackerCSRT_create,
        "kcf": cv2.TrackerKCF_create,
        "boosting": cv2.TrackerBoosting_create,
        "mil": cv2.TrackerMIL_create,
        "tld": cv2.TrackerTLD_create,
        "medianflow": cv2.TrackerMedianFlow_create,
        "mosse": cv2.TrackerMOSSE_create
    }
    loopIndex = 0
    fps = 0
    fpsCount=0
    st = time.time()
    k=0
    trackerList = []
    # -----------------Variable Editing Part-----------------------------------------------------------------
    seritCounts=[0,0,0,0]
    s1Kutu=[318,340,368,360]
    s2Kutu=[494,340,544,360]
    s3Kutu=[710,340,760,360]
    s4Kutu=[892,340,952,360]
    lineColor=(0, 0, 255)
    seritKutular=[s1Kutu,s2Kutu,s3Kutu,s4Kutu]
    detectionMod=10
    counted = (0, 0, 255)
    notCounted = (0, 255, 0)
    textColor = (51, 51, 255)
    # -----------------Variable Editing Part-----------------------------------------------------------------

    #out = cv2.VideoWriter(outputPath, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10,
                          #(frame_width, frame_height))
    while True:
        ret, frame = cap.read()
        fr.value = frame
        fpsCount += 1
        if (loopIndex == 0):
            st = time.time()
            loopIndex += 1



        index=0
        tm=time.time()
        for tr in trackerList:
            ye=tr[1][1]
            (success, boxes) = tr[0].update(frame)
            if abs(ye-boxes[0][1]) <2:
                tr[2]+=1
            if tr[2]==15:
                trackerList.remove(tr)
            else:

                trackerList[index][1]=[int(boxes[0][0]),int(boxes[0][1]),int(boxes[0][2]),int(boxes[0][3])]
                index+=1
                if tr[1][1]+(tr[1][3]/2)>yStart:
                    tBox = [tr[1][0], tr[1][1], tr[1][0] + tr[1][2], tr[1][1] + tr[1][3]]
                    boxIntersections=[]
                    for j in range(len(seritKutular)):
                        boxIntersections.append(kutuKarsilastir(seritKutular[j],tBox))
                    m=max(boxIntersections)
                    for j in range(len(boxIntersections)):
                        if(m==boxIntersections[j]):
                            seritCounts[j]+=1
                            break

                    trackerList.remove(tr)

                    index-=1

        print("trackT",time.time()-tm)
        if k%detectionMod==0:


            detStarter.value = True
            while not lock.value:
                nobody = 0

            realTimeCars = rtcars.value

            if len(trackerList)!=0:
                for i in range(len(realTimeCars[0])):
                    match = False
                    benzerCar=0
                    silinecekler=[]
                    j=0
                    for tr in trackerList:
                        rBox=[realTimeCars[0][i][0],realTimeCars[0][i][1],realTimeCars[0][i][2]+realTimeCars[0][i][0],realTimeCars[0][i][3]+realTimeCars[0][i][1]]
                        tBox=[tr[1][0],   tr[1][1], tr[1][0]+tr[1][2],tr[1][1]+tr[1][3]]
                        s=kutuKarsilastir(tBox,rBox)

                        if s>0.3:
                            match = True
                            trackers = cv2.MultiTracker_create()
                            tracker = OPENCV_OBJECT_TRACKERS["medianflow"]()
                            trackers.add(tracker, frame, (realTimeCars[0][i][0], realTimeCars[0][i][1], realTimeCars[0][i][2], realTimeCars[0][i][3]))
                            trackerList[j]=[trackers, [realTimeCars[0][i][0], realTimeCars[0][i][1], realTimeCars[0][i][2],realTimeCars[0][i][3]] ,0   ]

                        if s>0.8:
                            if benzerCar>0:
                                trackerList.remove(trackerList[j])
                                j-=1
                            benzerCar+=1
                        j+=1


                    if match==False:
                        trackers = cv2.MultiTracker_create()
                        tracker = OPENCV_OBJECT_TRACKERS["medianflow"]()
                        trackers.add(tracker, frame, (realTimeCars[0][i][0], realTimeCars[0][i][1], realTimeCars[0][i][2], realTimeCars[0][i][3]))
                        trackerList.append( [trackers, [realTimeCars[0][i][0], realTimeCars[0][i][1], realTimeCars[0][i][2],realTimeCars[0][i][3]] ,0   ]  )



            else:
                for i in range(len(realTimeCars[0])):

                    trackers = cv2.MultiTracker_create()
                    tracker = OPENCV_OBJECT_TRACKERS["medianflow"]()
                    trackers.add(tracker, frame, (realTimeCars[0][i][0],realTimeCars[0][i][1],realTimeCars[0][i][2],realTimeCars[0][i][3]))
                    trackerList.append(   [   trackers    ,   [   realTimeCars[0][i][0]  , realTimeCars[0][i][1]    ,   realTimeCars[0][i][2]  , realTimeCars[0][i][3]  ],0  ]    )
            lock.value=False

        k+=1
        for box in trackerList:
            (x, y, w, h) = [int(v) for v in box[1]]
            if box[2]==0:
                if y+h>yStart:
                   cv2.rectangle(frame, (x, y), (x + w, y + h), counted, 2)
                else:
                    cv2.rectangle(frame, (x, y), (x + w, y + h),notCounted , 2)
                text = " {}  ".format("car")
                cv2.putText(frame, text, (x + 10, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, textColor, 2)
        for j in range(len(seritKutular)):
            cv2.rectangle(frame, (seritKutular[j][0], seritKutular[j][1]), (seritKutular[j][2], seritKutular[j][3]), notCounted, 2)
        for j in range(len(seritCounts)):
            cv2.putText(frame, str(seritCounts[j]), (353+(200*j), 486), cv2.FONT_HERSHEY_SIMPLEX, 0.5, textColor, 2)
        cv2.putText(frame, "Ortalam FPS:" +str(int(fps)), (1075, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (51, 51, 255), 1)
        cv2.putText(frame, "Detection FPS:" + str(int(fp.value)), (597, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (51, 51, 255), 1)
        cv2.line(frame, (xStart, yStart), (xFinish, yStart), lineColor, 3)
        cv2.imshow('Stream',frame)
        cv2.waitKey(1)
        if time.time() - st > 5:
            fps=fpsCount / (time.time() - st)
            fpsCount = 0
            st = time.time()


if __name__ == '__main__':
    a = []
    fr = None
    manager = Manager()
    frameIndex = manager.Value("i", 0)
    tv = manager.Value("i", 0)
    fps = manager.Value("i", 0)
    lock = manager.Value(bool, False)
    lock2 = manager.Value(bool, False)
    f = manager.Value(memoryview, fr)
    f2 = manager.Value(memoryview, fr)
    detStarter = manager.Value(bool, False)
    modelLoaded = manager.Value(bool, False)
    rtCars = manager.Value(list, a)
    trackers = manager.Value(list, a)
    seritler = manager.Value(list, a)
    information = manager.Value(list, a)
    track = Process(target=tracking,
                    args=(rtCars, f, detStarter, modelLoaded, lock, trackers, lock2, seritler,fps))
    det = Process(target=detect, args=(rtCars, f, detStarter, modelLoaded, lock,fps))

    det.start()
    track.start()

    det.join()
    track.join()
