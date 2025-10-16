from mtcnn.mtcnn import MTCNN

def extract_whole_face(image_rgb):
    # Create an MTCNN detector instance
    detector = MTCNN()
    # Detect faces in the image
    faces = detector.detect_faces(image_rgb)
    try :
        # Assuming there's only one face in the image, or you can choose the largest face
        face = faces[0]
        # Get the bounding box coordinates of the face
        x, y, width, height = face['box']
        # Crop the whole face region from the image
        whole_face_img = image_rgb[y:y+height, x:x+width]
        return whole_face_img
    except:
        return image_rgb