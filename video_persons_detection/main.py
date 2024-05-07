from imageai.Detection import VideoObjectDetection
import os

if __name__ == "__main__":

    execution_path = os.getcwd() # путь к проекту

    detector = VideoObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath(os.path.join(execution_path , "retinanet_resnet50_fpn_coco-eeacb38b.pth"))
    detector.loadModel()

    custom_objects = detector.CustomObjects(person=True)

    video_path = detector.detectObjectsFromVideo(
        custom_objects=custom_objects,
        input_file_path=os.path.join(execution_path, "crowd.mp4"),
        output_file_path=os.path.join(execution_path, "crowd_person_detected"),
        frames_per_second=20, 
        log_progress=True
        )
    print(video_path)
