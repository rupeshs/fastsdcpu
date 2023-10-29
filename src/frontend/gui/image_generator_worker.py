from PyQt5.QtCore import (
    pyqtSlot,
    QRunnable,
    pyqtSignal,
    pyqtSlot,
)
from PyQt5.QtCore import QObject
import traceback
import sys


class WorkerSignals(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)


class ImageGeneratorWorker(QRunnable):
    def __init__(self, fn, *args, **kwargs):
        super(ImageGeneratorWorker, self).__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)
        finally:
            self.signals.finished.emit()
