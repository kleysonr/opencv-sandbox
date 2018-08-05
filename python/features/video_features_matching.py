import cv2
import imutils
import numpy as np
import time
from imutils.feature.factories import FeatureDetector_create, DescriptorExtractor_create, DescriptorMatcher_create

'''
    File name: video_features_matching.py
    Author: Kleyson Rios
    twitter: @kleysonr
    LinkedIn: https://www.linkedin.com/in/kleyson-rios-60347312/
    Python Version: 3.5
'''

class App:
    
    def __init__(self, src=0, detectorAlgo="FAST", extractorAlgo="SIFT", matcherAlgo="BruteForce"):
        self.camera = cv2.VideoCapture(src)
        self.first = True
        self.kpsImage = None
        self.detector = self.createDetector(algo=detectorAlgo)
        self.extractor = self.createExtractor(algo=extractorAlgo)
        self.matcher = self.createMatcher(algo=matcherAlgo)
        
    def createInitialImage(self, depth=3):
        return np.zeros([self.frameH, self.frameW, depth], dtype=np.uint8)
        
    # Options ['BRISK', 'DENSE', 'DOG', 'SIFT', 'FAST', 'FASTHESSIAN', 'SURF', 'GFTT', 'HARRIS', 'MSER', 'ORB', 'STAR']
    def createDetector(self, algo="FAST"):
        detector = None
        if algo == "DOG":
            detector = FeatureDetector_create("SIFT")
        elif algo == "FASTHESSIAN":
            detector = FeatureDetector_create("SURF")
        else:
            detector = FeatureDetector_create(algo)   
                 
        return detector
        
    # Options ['RootSIFT', 'SIFT', 'SURF']
    def createExtractor(self, algo="SIFT"):
        extractor = DescriptorExtractor_create(algo)
        return extractor
        
    # Options ['BruteForce', 'BruteForce-SL2', 'BruteForce-L1', 'FlannBased']
    def createMatcher(self, algo="BruteForce"):
        matcher = DescriptorMatcher_create(algo)
        return matcher        

    def run(self):
        
        # keep looping
        while True:
            
            stime = time.time()
            
            # grab the current frame
            (grabbed, frame) = self.camera.read()
         
            # if we are viewing a video and we did not grab a
            # frame, then we have reached the end of the video
            if not grabbed:
                break
         
            # resize image
            frame = imutils.resize(frame, width=600)
            
            # is the first frame
            if self.first:
                (self.frameH, self.frameW) = frame.shape[:2]
                self.vis = np.zeros((self.frameH, 2 * self.frameW, 3), dtype="uint8")
                self.image = self.createInitialImage()
                self.first = False
            
            # read pressed key
            key = cv2.waitKey(1) & 0xFF

            # if the 'space' key is pressed, take a snapshot and extract keypoints and features from the image/roi
            if key == ord(' '):
                cv2.destroyAllWindows()
                self.image = frame.copy()
                grayImage = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                
                (x, y, w, h) = cv2.selectROI("Select a ROI in the image - Press SPACE or ENTER to save or C to CANCEL ...", self.image, fromCenter=False, showCrosshair=False)
                
                mask = None
                if (x + y + w + h) > 0:
                    mask = self.createInitialImage(depth=1)
                    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
                    cv2.imshow('mask', mask)
                    
                    cv2.rectangle(self.image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                _kpsImage = self.detector.detect(grayImage, mask)
                (self.kpsImage, self.featuresImage) = self.extractor.compute(grayImage, _kpsImage)
                
                cv2.destroyAllWindows()
            
            # extract features from the frames and matches with the snapshot image
            matches = []
            if self.kpsImage is not None:
                grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                _kpsFrame = self.detector.detect(grayFrame, None)
                
                (self.kpsFrame, self.featuresFrame) = self.extractor.compute(grayFrame, _kpsFrame)
                
                rawMatches = self.matcher.knnMatch(self.featuresImage, self.featuresFrame, k=2)
            
                for m in rawMatches:
                    # ensure the distance passes David Lowe's ratio test
                    if len(m) == 2 and m[0].distance < m[1].distance * 0.8:
                        matches.append((m[0].trainIdx, m[0].queryIdx))       
        
                print("# keypoints image: {} / # keypoints frame: {} / # of matched keypoints: {}".format(len(self.kpsImage), len(self.kpsFrame), len(matches)))

            # if the 'q' key is pressed, stop the loop
            if key == ord("q"):
                break

            # do a kind of np.hstack()
            self.vis[0:self.frameH, 0:self.frameW] = self.image
            self.vis[0:self.frameH, self.frameW:] = frame
            
            # loop over the matches
            for (trainIdx, queryIdx) in matches:
                
                # generate a random color and draw the match
                color = np.random.randint(0, high=255, size=(3,))
                color = tuple(map(int, color))
                ptA = (int(self.kpsImage[queryIdx].pt[0]), int(self.kpsImage[queryIdx].pt[1]))
                ptB = (int(self.kpsFrame[trainIdx].pt[0] + self.frameW), int(self.kpsFrame[trainIdx].pt[1]))
                cv2.line(self.vis, ptA, ptB, color, 2)
            
                
            print('FPS {:.1f}'.format(1 / (time.time() - stime)))
                 
            cv2.imshow("Image / Frame", self.vis)

        # clean up the camera and close any open windows 
        self.camera.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':

    # detectorAlgo  Options: ['BRISK', 'DENSE', 'DOG', 'SIFT', 'FAST', 'FASTHESSIAN', 'SURF', 'GFTT', 'HARRIS', 'MSER', 'ORB', 'STAR']
    # extractorAlgo Options: ['RootSIFT', 'SIFT', 'SURF']
    # matcherAlgo   Options: ['BruteForce', 'BruteForce-SL2', 'BruteForce-L1', 'FlannBased']

    App(src=0, detectorAlgo="SURF", extractorAlgo="RootSIFT", matcherAlgo="BruteForce").run()
