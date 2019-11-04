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
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QPushButton, QDialog, QMessageBox,
                             QVBoxLayout, QHBoxLayout,
                             QLabel, QPlainTextEdit, QListView,
                             )


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


class Results_dialog(QDialog):
    """
    widget for visualizing text output
    """
    def __init__(self):
        super().__init__()

        self.setWindowTitle("")

        hbox = QVBoxLayout()

        self.lb = QLabel("")
        hbox.addWidget(self.lb)

        self.ptText = QPlainTextEdit()
        hbox.addWidget(self.ptText)

        hbox2 = QHBoxLayout()
        self.pbSave = QPushButton("Save text")
        self.pbSave.clicked.connect(self.save_results)
        hbox2.addWidget(self.pbSave)

        self.pbCancel = QPushButton("Cancel")
        self.pbCancel.clicked.connect(self.reject)
        hbox2.addWidget(self.pbCancel)
        self.pbCancel.setVisible(False)

        self.pbOK = QPushButton("OK")
        self.pbOK.clicked.connect(self.accept)
        hbox2.addWidget(self.pbOK)

        hbox.addLayout(hbox2)

        self.setLayout(hbox)

        self.resize(540, 640)


    def save_results(self):
        """
        save content of self.ptText
        """

        fn = QFileDialog().getSaveFileName(self, "Save results", "", "Text files (*.txt *.tsv);;All files (*)")
        file_name = fn[0] if type(fn) is tuple else fn

        if file_name:
            try:
                with open(file_name, "w") as f:
                    f.write(self.ptText.toPlainText())
            except Exception:
                QMessageBox.critical(self, "DORIS", "The file {} can not be saved".format(file_name))


