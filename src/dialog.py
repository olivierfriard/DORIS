#!/usr/bin/env python3
"""
DORIS
Detection of Objects Research Interactive Software
Copyright 2017-2019 Olivier Friard

This file is part of DORIS.

  DORIS is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 3 of the License, or
  any later version.

  DORIS is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program; if not see <http://www.gnu.org/licenses/>.

"""

import sys
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *


def MessageDialog(title, text, buttons):
    response = ""
    message = QMessageBox()
    message.setWindowTitle(title)
    message.setText(text)
    message.setIcon(QMessageBox.Question)
    for button in buttons:
        message.addButton(button, QMessageBox.YesRole)

    message.setWindowFlags(Qt.WindowStaysOnTopHint)
    message.exec_()
    return message.clickedButton().text()


class CheckListWidget(QDialog):
    """
    widget for visualizing list with check boxes
    """
    def __init__(self, items):
        super().__init__()

        self.setWindowTitle("")
        self.checked = []

        hbox = QVBoxLayout()

        self.lb = QLabel("")
        hbox.addWidget(self.lb)

        self.lv = QListView()
        hbox.addWidget(self.lv)

        hbox2 = QHBoxLayout()

        self.pbCancel = QPushButton("Cancel")
        self.pbCancel.clicked.connect(self.reject)
        hbox2.addWidget(self.pbCancel)

        self.pbOK = QPushButton("OK")
        self.pbOK.clicked.connect(self.ok)
        hbox2.addWidget(self.pbOK)

        hbox.addLayout(hbox2)

        self.setLayout(hbox)

        self.model = QStandardItemModel(self.lv)

        for string in items:
            item = QStandardItem(string)
            item.setCheckable(True)
            self.model.appendRow(item)

        self.lv.setModel(self.model)


    def ok(self):
        self.checked = []
        i = 0
        while self.model.item(i):
            if self.model.item(i).checkState():
                self.checked.append(self.model.item(i).text())
            i += 1
        self.accept()


'''
class Input_dialog(QDialog):
    """
    dialog for user input. Elements can be checkbox, lineedit, spinbox

    """

    def __init__(self, label_caption, elements_list):
        super().__init__()

        hbox = QVBoxLayout()
        self.label = QLabel()
        self.label.setText(label_caption)
        hbox.addWidget(self.label)

        self.elements = {}
        for element in elements_list:
            if element[0] == "cb":
                self.elements[element[1]] = QCheckBox(element[1])
                self.elements[element[1]].setChecked(element[2])
                hbox.addWidget(self.elements[element[1]])
            if element[0] == "le":
                lb = QLabel(element[1])
                hbox.addWidget(lb)
                self.elements[element[1]] = QLineEdit()
                hbox.addWidget(self.elements[element[1]])
            if element[0] == "sb":
                lb = QLabel(element[1])
                hbox.addWidget(lb)
                self.elements[element[1]] = QSpinBox()
                self.elements[element[1]].setRange(element[2], element[3])
                self.elements[element[1]].setSingleStep(element[4])
                self.elements[element[1]].setValue(element[5])
                hbox.addWidget(self.elements[element[1]])

        hbox2 = QHBoxLayout()

        self.pbCancel = QPushButton("Cancel")
        self.pbCancel.clicked.connect(self.reject)
        hbox2.addWidget(self.pbCancel)

        self.pbOK = QPushButton("OK")
        self.pbOK.clicked.connect(self.accept)
        self.pbOK.setDefault(True)
        hbox2.addWidget(self.pbOK)

        hbox.addLayout(hbox2)

        self.setLayout(hbox)

        self.setWindowTitle("title")
'''


