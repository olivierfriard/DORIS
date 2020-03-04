"""
DORIS
Detection of Objects Research Interactive Software
Copyright 2017-2020 Olivier Friard

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

import os
from doris import version
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QPushButton, QDialog, QMessageBox,
                             QVBoxLayout, QHBoxLayout,
                             QLabel, QPlainTextEdit, QListView,
                             QFileDialog
                             )


def MessageDialog(title, text, buttons):

    message = QMessageBox()
    message.setWindowTitle(title)
    message.setText(text)
    message.setIcon(QMessageBox.Question)
    for button in buttons:
        message.addButton(button, QMessageBox.YesRole)

    message.setWindowFlags(Qt.WindowStaysOnTopHint)
    message.exec_()
    return message.clickedButton().text()


def error_info(exc_info: tuple) -> tuple:
    """
    return details about error
    usage: error_info(sys.exc_info())

    Args:
        sys.exc_info() (tuple):

    Returns:
        tuple: error type, error file name, error line number
    """

    exc_type, exc_obj, exc_tb = exc_info
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    return (exc_obj, fname, exc_tb.tb_lineno)


def error_message(task: str, exc_info: tuple) -> None:
    """
    show details about the error

    """
    error_type, error_file_name, error_lineno = error_info(exc_info)
    QMessageBox.critical(None, "DORIS",
                         (f"An error occured during {task}.<br>"
                          f"DORIS version: {version.__version__}<br>"
                          f"Error: {error_type}<br>"
                          f"in {error_file_name} "
                          f"at line # {error_lineno}<br><br>"
                          "Please report this problem to improve the software at:<br>"
                          '<a href="https://github.com/olivierfriard/DORIS/issues">https://github.com/olivierfriard/DORIS/issues</a>'
                          ))

    return (error_type, error_file_name, error_lineno)


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


