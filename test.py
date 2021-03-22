from threading import Thread
import cv2
from lib_detection import *
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
                if self.get:
                    self.reader_queue.put(self.image)
                self.get = not self.get
 
class Detector(Thread):
    def __init__(self, reader_queue, detector_queue):
        super().__init__()
        self.reader_queue = reader_queue
        self.detector_queue = detector_queue
        self.boxes = []
        self.confidences = []
        self.class_ids = []
       
    def run(self):
        while True:
            if not self.reader_queue.empty():
                frame = self.reader_queue.get()
                boxes, confidences, class_ids = Detect_person(frame = frame, boxes = self.boxes, confidences = self.confidences, class_ids = self.class_ids)
                indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
                font = cv2.FONT_HERSHEY_PLAIN
                for i in range(len(boxes)):
                    if i in indexes:
                        x, y, w, h = boxes[i]
                        label = str(classes[class_ids[i]])
                        color = (0, 0, 255)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(frame, label, (x, y + 30), font, 2, color, 2)
                self.detector_queue.put(frame)
                remove_list(self.boxes)
                remove_list(self.confidences)
                remove_list(self.class_ids)


class Writer(Thread):
    def __init__(self, detector_queue):
        super().__init__()
        self.detector_queue = detector_queue
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter('output.avi', self.fourcc, 5, (768, 576))

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
