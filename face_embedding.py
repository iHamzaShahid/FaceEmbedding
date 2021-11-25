from retinaface import RetinaFace
import os
import json
import numpy as np
from deepface import DeepFace

class Error(Exception):
    """Base class for other exceptions"""
    pass

class PathNotFound(Error):
    pass

class MultipleFaceDetected(Error):
    pass

class InvalidPose(Error):
    pass

class NoFaceDetected(Error):
    pass

class SmallFaceDetected(Error):
    pass


def Face_Emb(image_path , name , person_id):
    file_path = os.path.dirname(os.path.abspath(__file__)) + os.sep
    diction = {'names':[], 'embeddings':[]}
    
    try:
        if os.path.exists(image_path) ==False:
            raise PathNotFound
        else:
            faces = RetinaFace.extract_faces(img_path = image_path, align = True)
            faces = np.array(faces)
            

            # Checking if multiple faces detected
            if len(faces) > 1:
                raise MultipleFaceDetected

            # Checking if no face detected
            if len(faces) < 1:
                raise NoFaceDetected

            # Checking if the height and width of detected face is less than 100 pixels
            if faces.shape[1] < 100 or faces.shape[2] < 100:
                raise SmallFaceDetected


            result = DeepFace.represent(image_path, model_name = 'ArcFace', model = None, enforce_detection = False, detector_backend = 'retinaface', align = True, normalization = 'ArcFace')
            name = name.replace('_', ' ')
            diction = {'names': name, 'embeddings': result}
            with open('dictionary.json', 'w') as handle:
                json.dump(diction, handle)
                                

                
    except PathNotFound:
        print("Check input path")
    except MultipleFaceDetected:
        print("Multiple face detected")
    except NoFaceDetected:
        print("No face Detected")
    except SmallFaceDetected:
        print("Detected face is smaller than requiured for embedding")
    except InvalidPose:
        print("Pose is not valid")
if __name__ == '__main__':
    Face_Emb("./images/1.jpg" , "name" , 1)
    

