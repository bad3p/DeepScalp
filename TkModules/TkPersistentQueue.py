
import os
from TkModules.TkIO import TkIO

#------------------------------------------------------------------------------------------------------------------------
# Persistent queue, located in file system
# Initialized using given init_callback returning list of queued items
#------------------------------------------------------------------------------------------------------------------------

class TkPersistentQueue():

    def __init__(self, path : str, init_callback):

        self._path = path

        if init_callback == None:
            raise RuntimeError('No init callback provided!')

        if not os.path.isfile(path):
            self._queue = init_callback()
            if not type(self._queue) is list:
                raise RuntimeError('Init callback expected to return list of queued items!')
            TkIO.write_at_path( path, self._queue )
        else:
            self._queue = TkIO.read_at_path( path )[0]

    def pop(self):
        result = self._queue[0]
        del self._queue[0]
        return result

    def push(self,val):
        self._queue.append(val)

    def flush(self):
        TkIO.write_at_path( self._path, self._queue )