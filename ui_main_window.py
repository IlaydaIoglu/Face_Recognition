from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(660, 530)
        self.horizontalLayout = QtWidgets.QHBoxLayout(Form)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.image_label = QtWidgets.QLabel(Form)
        self.image_label.setObjectName("image_label")
        self.verticalLayout.addWidget(self.image_label)
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.control_bt = QtWidgets.QPushButton(Form)
        self.control_bt.setObjectName("control_bt")
        self.verticalLayout.addWidget(self.control_bt)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.save_bt = QtWidgets.QPushButton(Form)
        self.save_bt.setObjectName("save_bt")
        self.verticalLayout.addWidget(self.save_bt)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.extract_bt = QtWidgets.QPushButton(Form)
        self.extract_bt.setObjectName("extract_bt")
        self.verticalLayout.addWidget(self.extract_bt)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.train_bt = QtWidgets.QPushButton(Form)
        self.train_bt.setObjectName("train_bt")
        self.verticalLayout.addWidget(self.train_bt)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.delete_bt = QtWidgets.QPushButton(Form)
        self.delete_bt.setObjectName("delete_bt")
        self.verticalLayout.addWidget(self.delete_bt)
        self.horizontalLayout.addLayout(self.verticalLayout)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Face Camera"))
        self.image_label.setText(_translate("Form", "Camera"))
        self.control_bt.setText(_translate("Form", "Start"))
        self.save_bt.setText(_translate("Form", "Save"))
        self.extract_bt.setText(_translate("Form", "Extract"))
        self.train_bt.setText(_translate("Form", "Train"))
        self.delete_bt.setText(_translate("Form", "Delete Temp"))


