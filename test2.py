from threading import Thread
import cv2
from lib_detection import *
from lib_detect_objectBS import *
import time
from queue import Queue

class Reader(Thread):
    def __init__(self,filePath, reader_queue):
        super().__init__()
        self.cap = cv2.VideoCapture(filePath)
        self.reader_queue = reader_queue
        #use variable "get" to reduce number of frame to be detected --> ensure realtime
        self.get = True

    def run(self):        
        while True:
            if self.cap.isOpened():
                (self.grabbed, self.image) = self.cap.read()
                self.reader_queue.put(self.image)
                #if self.get:
                    #self.reader_queue.put(self.image)
                #self.get = not self.get
 
class Detector(Thread):
    def __init__(self, reader_queue, detector_queue):
        super().__init__()
        self.reader_queue = reader_queue
        self.detector_queue = detector_queue
        self.bkg_list = []
       
    def run(self):
        while True:
            if not self.reader_queue.empty():
                frame = self.reader_queue.get()
                bkg_list = backgroundSubtraction(frame,self.bkg_list)
                #draw bounding box
                for va in bkg_list:
                    cv2.rectangle(frame, (va[0],va[1]), (va[0]+va[2],va[1]+va[3]), (255,0,0), 2, 1)
                self.detector_queue.put(frame)
                remove_list(self.bkg_list) 



class Writer(Thread):
    def __init__(self, detector_queue):
        super().__init__()
        self.detector_queue = detector_queue
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter('output.avi', self.fourcc, 10, (768, 576))

    def run(self):
        while True:
            if not self.detector_queue.empty():
                frame = self.detector_queue.get()
                self.out.write(frame)
                cv2.imshow('Detection', frame)
                if (cv2.waitKey(10) & 0xFF) == ord("q"):
                    break

def main():

    reader_queue = Queue()
    detector_queue = Queue()
    filePath = "1.avi"

    reader = Reader(filePath, reader_queue)
    reader.start()
    detector = Detector(reader.reader_queue, detector_queue)
    detector.start()
    writer = Writer(detector.detector_queue)
    writer.start()


if __name__ == "__main__":
    main()
