import glob
from natsort import natsorted

class ReadData(object):
    def __init__(self):
        self.path = "data/"
        self.path_len = len(self.path)
        self.config_path = glob.glob(self.path+"*")
    
    def org_files(self, data_dir):
        """Organise raw outputs based on configuration"""

        file_arr = [[ ] for i in range(len(self.config_path))]
        for config_ in range(len(file_arr)):
            file_arr[config_] = natsorted(glob.glob(self.config_path[config_]+"/"+data_dir+"/*"))

        return file_arr
