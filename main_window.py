# import system module
import sys
import os

# import some PyQt5 modules
from PyQt5.QtWidgets import QApplication, QWidget, QMessageBox
from PyQt5.QtGui import QImage, QIcon, QPixmap
from PyQt5.QtCore import QTimer

from tkinter import simpledialog, Tk
from ui_main_window import *
from face_lib import *
from worker import *



class MainWindow(QWidget):
    # class constructor
    def __init__(self):
        # call QWidget constructor
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        # icon
        self.setWindowIcon(QIcon('icon.png'))
        # create a timer
        self.viewTimer = QTimer()
        self.saveTimer = QTimer()
        # set timer timeout callback function
        self.viewTimer.timeout.connect(self.view_cam)
        self.saveTimer.timeout.connect(self.save_cam)
        # set control_bt callback clicked  function
        self.ui.control_bt.clicked.connect(self.control_view)
        self.ui.save_bt.clicked.connect(self.control_save)
        self.ui.train_bt.clicked.connect(self.train_thread)
        self.ui.extract_bt.clicked.connect(self.extract_thread)
        self.ui.delete_bt.clicked.connect(self.delete_thread)
        # waring message box
        self.msg = QMessageBox()
        self.msg.setWindowIcon(QIcon('warning.png'))
        # create face recognition
        self.face_rec = face_lib()
        # drone ip
        self.ip = 'tcp://192.168.1.1:5555'
        # prediction process
        self.process_this_frame = 16
        self.predictions = ""
        # folders and files
        self.video_base = 'video/'
        self.photo_base = 'photo/'
        self.face_base = 'faces/'
        self.train_file = "trained_knn_model.clf"
        # tkinter for questions
        self.root = Tk()
        self.root.withdraw()
        # create folders for first open
        if not os.path.exists(self.video_base) & os.path.exists(self.photo_base) & os.path.exists(self.face_base):
            os.mkdir(self.video_base)
            os.mkdir(self.photo_base)
            os.mkdir(self.face_base)
        # create thread pool for max thread
        self.pool = QThreadPool()

    # # start/stop viewing
    def control_view(self):
        # if timer is stopped
        if self.saveTimer.isActive():
            self.msg.setWindowTitle("Warning")
            self.msg.setText("Please stop recording")
            self.msg.exec_()
        if not self.viewTimer.isActive():
            if not self.saveTimer.isActive():
                # create video capture
                self.cap = cv2.VideoCapture(0)
                # start timer
                self.viewTimer.start(30)
                # update control_bt text
                self.ui.control_bt.setText("Stop")
        # if timer is started
        else:
            # stop timer
            self.viewTimer.stop()
            # release video capture
            self.cap.release()
            # update control_bt text
            self.ui.control_bt.setText("Start")
            self.ui.image_label.setText("Camera")

    # start/stop saving
    def control_save(self):
        # if camera timer active
        if self.viewTimer.isActive():
            self.msg.setWindowTitle("Warning")
            self.msg.setText("Please stop the camera")
            self.msg.exec_()
        # if timer is stopped
        if not self.saveTimer.isActive():
            if not self.viewTimer.isActive():
                # create video capture
                self.cap = cv2.VideoCapture(0)
                # set video output
                dirname = simpledialog.askstring(title="Name", prompt="Write person name will be saved")
                # Parent Directory path
                if not dirname is None:
                    path = os.path.join(self.video_base, dirname)
                    if not os.path.exists(path):
                        os.mkdir(path)
                    self.out = cv2.VideoWriter(f'{self.video_base}{dirname}/person.avi',
                                               cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 24, (640, 480))
                    # start timer
                    self.saveTimer.start(30)
                    # update save_bt text
                    self.ui.save_bt.setText("Stop")
                else:
                    pass
        # if timer is started
        else:
            # stop timer
            self.saveTimer.stop()
            # release video capture
            self.cap.release()
            # update save_bt text
            self.ui.save_bt.setText("Save")
            self.ui.image_label.setText("Camera")

    # view camera
    def view_cam(self):
        if not self.saveTimer.isActive():
            # read image in BGR format
            ret, image = self.cap.read()
            if ret:
                if os.path.isfile(self.train_file):
                    img = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
                    # process predictions
                    self.process_this_frame += 1
                    if self.process_this_frame % 30 == 0:
                        self.predictions = self.face_rec.predict(img, model_path=self.train_file)
                    image = self.face_rec.show_prediction_labels_on_image(image, self.predictions)
                # get image infos
                height, width, channel = image.shape
                step = channel * width
                # create QImage from image
                qImg = QImage(image.data, width, height, step, QImage.Format_BGR888)
                # show image in img_label
                self.ui.image_label.setPixmap(QPixmap.fromImage(qImg))

    # save video
    def save_cam(self):
        if not self.viewTimer.isActive():
            ret, image = self.cap.read()
            # get image infos
            height, width, channel = image.shape
            step = channel * width
            # write video
            self.out.write(image)
            # create QImage from image
            qImg = QImage(image.data, width, height, step, QImage.Format_BGR888)
            # show image in img_label
            self.ui.image_label.setPixmap(QPixmap.fromImage(qImg))

    def training(self):
        self.face_rec.train(self.face_base, model_save_path=self.train_file, n_neighbors=2)

    def extraction(self):
        c = Cropper(width=240, height=240, face_percent=75)
        self.face_rec.set_photo(self.video_base, self.photo_base)
        self.face_rec.set_faces(self.photo_base, self.face_base, c)

    def deletion(self):
        self.face_rec.delete_temp(self.photo_base)
        # self.face_rec.delete_temp(self.video_base)

    def extract_thread(self):
        worker = Worker(self.extraction)
        self.button_disable()
        worker.signals.finished.connect(self.button_enable)
        # Execute
        self.pool.start(worker)

    def train_thread(self):
        worker = Worker(self.training)
        self.button_disable()
        worker.signals.finished.connect(self.button_enable)
        # Execute
        self.pool.start(worker)

    def delete_thread(self):
        worker = Worker(self.deletion())
        self.button_disable()
        worker.signals.finished.connect(self.button_enable)
        # Execute
        self.pool.start(worker)

    def button_enable(self):
        self.ui.extract_bt.setEnabled(True)
        self.ui.train_bt.setEnabled(True)
        self.ui.delete_bt.setEnabled(True)

    def button_disable(self):
        self.ui.extract_bt.setEnabled(False)
        self.ui.train_bt.setEnabled(False)
        self.ui.delete_bt.setEnabled(False)


if __name__ == '__main__':
    app = QApplication(sys.argv)

    # create and show mainWindow
    mainWindow = MainWindow()
    mainWindow.show()

    sys.exit(app.exec_())
