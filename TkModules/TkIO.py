
import pickle
import os.path
from typing import IO

#------------------------------------------------------------------------------------------------------------------------
# Data serialization methods
#------------------------------------------------------------------------------------------------------------------------    

class TkIO():

    #------------------------------------------------------------------------------------------------------------------------    
    # Writes arbitrary object to the file at the current position
    #------------------------------------------------------------------------------------------------------------------------    

    @staticmethod
    def write_to_file(f:IO, object):
        objectBytes = pickle.dumps( object )
        chunkLength = len( objectBytes )
        f.write( chunkLength.to_bytes( 4, byteorder='big', signed=False ) )
        f.write( objectBytes )

    #------------------------------------------------------------------------------------------------------------------------    
    # Reads arbitrary object written at the current file position
    # Raises exception if object is unable to be read
    #------------------------------------------------------------------------------------------------------------------------            

    @staticmethod
    def read_from_file(f:IO):
        headerChuck = f.read(4)
        if headerChuck == b"":
            raise RuntimeError('End of stream')
        chunkLength = int.from_bytes( headerChuck, byteorder='big', signed=False )
        objectBytes = f.read(chunkLength)
        if objectBytes == b"":
            raise RuntimeError('Corrupted data stream')
        object = pickle.loads(objectBytes)
        return object

    #------------------------------------------------------------------------------------------------------------------------
    # Reads arbitrary object written at the current file position
    # Variant that will raise no exceptions, instead returning success flag as a second item in tuple
    #------------------------------------------------------------------------------------------------------------------------
    
    @staticmethod
    def read_from_file_noex(f:IO):
        headerChuck = f.read(4)
        if headerChuck == b"":
            return None, False
        chunkLength = int.from_bytes( headerChuck, byteorder='big', signed=False )
        objectBytes = f.read(chunkLength)
        if objectBytes == b"":
            return None, False
        object = pickle.loads(objectBytes)
        return object, True

    #------------------------------------------------------------------------------------------------------------------------
    # Writes arbitrary object at the newly created file provided by path
    #------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def write_at_path(filename : str, object):
        with open( filename, 'wb+') as file:
            file.seek( 0, 0 )
            objectBytes = pickle.dumps( object )
            chunkLength = len( objectBytes )
            file.write( chunkLength.to_bytes( 4, byteorder='big', signed=False ) )
            file.write( objectBytes )

    #------------------------------------------------------------------------------------------------------------------------
    # Writes arbitrary object at the end of the file provided by path
    #------------------------------------------------------------------------------------------------------------------------

    @staticmethod            
    def append_at_path(filename : str, object):
        with open( filename, 'ab+') as file:
            file.seek( 0, 2 )
            objectBytes = pickle.dumps( object )
            chunkLength = len( objectBytes )
            file.write( chunkLength.to_bytes( 4, byteorder='big', signed=False ) )
            file.write( objectBytes )

    #------------------------------------------------------------------------------------------------------------------------
    # Reads the content of file provided by path
    # Optional index argument provides subset of positions to be read (for further info, see index_at_path) 
    # The content is returned in the form of the list of arbitrary objects.
    #------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def read_at_path(filename : str, index=None):
        if not os.path.isfile(filename):
            return []
        result = []
        with open( filename, 'rb') as file:
            if index == None:
                file.seek( 0, 0 )
                while True:     
                    headerChuck = file.read(4)
                    if headerChuck == b"":
                        break # end of file
                    chunkLength = int.from_bytes( headerChuck, byteorder='big', signed=False )
                    objectBytes = file.read(chunkLength)
                    if objectBytes == b"":
                        raise RuntimeError('Corrupted data stream')
                    object = pickle.loads(objectBytes)
                    result.append(object)
            else:
                for position in index:
                    file.seek( position, 0 )
                    headerChuck = file.read(4)
                    if headerChuck == b"":
                        raise RuntimeError('Invalid index data')
                    chunkLength = int.from_bytes( headerChuck, byteorder='big', signed=False )
                    objectBytes = file.read(chunkLength)
                    if objectBytes == b"":
                        raise RuntimeError('Corrupted data stream')
                    object = pickle.loads(objectBytes)
                    result.append(object)

        return result

    #------------------------------------------------------------------------------------------------------------------------
    # Returns the index of the content of the file provided by path
    # The index is the list of positions of individual arbitrary objects
    #------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def index_at_path(filename : str):
        if not os.path.isfile(filename):
            return []
        
        index = []

        with open( filename, 'rb') as file:
            file.seek( 0, 0 )   
            while True:        
                filePos = file.tell()
                headerChuck = file.read(4)
                if headerChuck == b"":
                    break # end of file
                index.append( filePos )
                chunkLength = int.from_bytes( headerChuck, byteorder='big', signed=False )
                file.seek( chunkLength, 1 )

        return index
