import os
from autocrop import Cropper
from PIL import Image, ImageDraw
import math
from sklearn import neighbors
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import cv2
import pickle
import numpy as np
import shutil

class face_lib(object):

    def set_photo(self, video_base, photo_base):
        video_dir = os.listdir(video_base)
        for label in video_dir:
            for i, fn in enumerate(os.listdir(os.path.join(video_base, label))):
                print(f"start collecting photos from {label}'s data")
                cap = cv2.VideoCapture(os.path.join(video_base, label, fn))
                frame_count = 0
                while True:
                    # read video frame
                    ret, raw_img = cap.read()
                    # process every 5 frames
                    if frame_count % 5 == 0 and raw_img is not None:
                        h, w, _ = raw_img.shape
                        path = os.path.join(photo_base, label)
                        if not os.path.exists(path):
                            os.mkdir(path)
                        cv2.imwrite(f'{photo_base}{label}/{frame_count}.jpg', raw_img)

                    frame_count += 1
                    if not cap.get(cv2.CAP_PROP_FRAME_COUNT) == 0: # sometimes frame count returns 0
                        if frame_count == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                            break
                    elif frame_count == 500:
                        break

    def set_faces(self, photo_base, face_base, cropper):
        photo_dir = os.listdir(photo_base)
        for label in photo_dir:
            print(f"start collecting faces from {label}'s data")
            for i, fn in enumerate(os.listdir(os.path.join(photo_base, label))):

                photos = os.path.join(photo_base, label, fn)
                try:
                    cropped_array = cropper.crop(photos)
                except (AttributeError, TypeError):
                    pass
                if cropped_array is not None:
                    faces = Image.fromarray(cropped_array)
                path = os.path.join(face_base, label)
                if not os.path.exists(path):
                    os.mkdir(path)
                faces.save(f'{face_base}{label}/{i}.jpg')

    def train(self, train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
        """
        Trains a k-nearest neighbors classifier for face recognition.

        :param train_dir: directory that contains a sub-directory for each known person, with its name.

         (View in source code to see train_dir example tree structure)

         Structure:
            <train_dir>/
            ├── <person1>/
            │   ├── <somename1>.jpeg
            │   ├── <somename2>.jpeg
            │   ├── ...
            ├── <person2>/
            │   ├── <somename1>.jpeg
            │   └── <somename2>.jpeg
            └── ...

        :param model_save_path: (optional) path to save model on disk
        :param n_neighbors: (optional) number of neighbors to weigh in classification. Chosen automatically if not specified
        :param knn_algo: (optional) underlying data structure to support knn.default is ball_tree
        :param verbose: verbosity of training
        :return: returns knn classifier that was trained on the given data.
        """
        X = []
        y = []

        # Loop through each person in the training set
        for class_dir in os.listdir(train_dir):
            if not os.path.isdir(os.path.join(train_dir, class_dir)):
                continue

            # Loop through each training image for the current person
            for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
                image = face_recognition.load_image_file(img_path)
                face_bounding_boxes = face_recognition.face_locations(image)

                if len(face_bounding_boxes) != 1:
                    # If there are no people (or too many people) in a training image, skip the image.
                    if verbose:
                        print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(
                            face_bounding_boxes) < 1 else "Found more than one face"))
                else:
                    # Add face encoding for current image to the training set
                    X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                    y.append(class_dir)

        # Determine how many neighbors to use for weighting in the KNN classifier
        if n_neighbors is None:
            n_neighbors = int(round(math.sqrt(len(X))))
            if verbose:
                print("Chose n_neighbors automatically:", n_neighbors)

        # Create and train the KNN classifier
        knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
        knn_clf.fit(X, y)

        # Save the trained KNN classifier
        if model_save_path is not None:
            with open(model_save_path, 'wb') as f:
                pickle.dump(knn_clf, f)

        return knn_clf

    def predict(self, X_frame, knn_clf=None, model_path=None, distance_threshold=0.5):
        """
        Recognizes faces in given image using a trained KNN classifier

        :param X_frame: frame to do the prediction on.
        :param knn_clf: (optional) a knn classifier object. if not specified, model_save_path must be specified.
        :param model_path: (optional) path to a pickled knn classifier. if not specified, model_save_path must be knn_clf.
        :param distance_threshold: (optional) distance threshold for face classification. the larger it is, the more chance
               of mis-classifying an unknown person as a known one.
        :return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
            For faces of unrecognized persons, the name 'unknown' will be returned.
        """
        if knn_clf is None and model_path is None:
            raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

        # Load a trained KNN model (if one was passed in)
        if knn_clf is None:
            with open(model_path, 'rb') as f:
                knn_clf = pickle.load(f)

        X_face_locations = face_recognition.face_locations(X_frame)

        # If no faces are found in the image, return an empty result.
        if len(X_face_locations) == 0:
            return []

        # Find encodings for faces in the test image
        faces_encodings = face_recognition.face_encodings(X_frame, known_face_locations=X_face_locations)

        # Use the KNN model to find the best matches for the test face
        closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
        are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

        # Predict classes and remove classifications that aren't within the threshold
        return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in
                zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]

    def show_prediction_labels_on_image(self, frame, predictions):
        """
        Shows the face recognition results visually.

        :param frame: frame to show the predictions on
        :param predictions: results of the predict function
        :return opencv suited image to be fitting with cv2.imshow fucntion:
        """
        pil_image = Image.fromarray(frame)
        draw = ImageDraw.Draw(pil_image)

        for name, (top, right, bottom, left) in predictions:
            # enlarge the predictions for the full sized image.
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2
            # Draw a box around the face using the Pillow module
            draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

            # There's a bug in Pillow where it blows up with non-UTF-8 text
            # when using the default bitmap font
            name = name.encode("UTF-8")
            # Draw a label with a name below the face
            text_width, text_height = draw.textsize(name)
            draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
            draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

        # Remove the drawing library from memory as per the Pillow docs.
        del draw
        # Save image in open-cv format to be able to show it.

        opencvimage = np.array(pil_image)
        return opencvimage

    def delete_temp(self, folder):
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
