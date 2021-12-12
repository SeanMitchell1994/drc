import os
import datetime
import time

class Logger:
    def __init__(self):
        #print("logger init")

        self.mode = ""
        self.dir_name = ""
        self.file_name = ""
        self.time = ""
        self.timestamp = ""

        self.Setup()

    def Setup(self):
        print("Path setup")

        # Checks if the output path exists and makes it if it doesn't
        try:
            os.makedirs("../../run/")
        except FileExistsError:
            # path already exists, move on
            pass

        try:
            os.makedirs("../../run/output")
        except FileExistsError:
            # path already exists, move on
            pass
    
        ts = time.time()
        self.st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H-%M-%S')
        try:
            os.makedirs("../../run/output/%s" % self.st)
        except FileExistsError:
            # path already exists, move on
            pass

        self.dir_name = "../../run/output/%s" % self.st
        try:
            os.makedirs("%s/metrics" % self.dir_name)
        except FileExistsError:
            # path already exists, move on
            pass

        try:
            os.makedirs("%s/plots" % self.dir_name)
        except FileExistsError:
            # path already exists, move on
            pass

        try:
            os.makedirs("%s/data" % self.dir_name)
        except FileExistsError:
            # path already exists, move on
            pass

    def Log(self):
        print("logging function")

    def Save_Data(self, data, fname):
        #print("Data save")
        data.to_csv("%s/data/%s.csv" % (self.dir_name, fname))

    def Save_Fig(self,plt, legend, fname):
        #print("Fig save")
        plt.savefig("%s/plots/%s.jpg" % (self.dir_name, fname), bbox_extra_artists=(legend,), bbox_inches='tight', dpi=100)