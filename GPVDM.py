import os
import shutil
import sys
import pandas
import pandas as pd
import ujson as json
import time
import func_timeout

sys.path.append("/home/paul/Oghma/oghma/gui/")

from oghma_api import oghma_api
from json_root import json_root

class gpvdm:
    def __init__(self):

        self.api = oghma_api(verbose=False)

        self.script_name = os.path.basename(__file__).split('.')[0]
        self.scan_dir = os.path.join(os.getcwd(), self.script_name)

        self.api.mkdir(self.scan_dir)
        self.api.server.server_base_init(self.scan_dir)

    def create_job(self, sim_name,material):
        material = material.split('MAT')[1][:-1]
        self.sim_path = os.path.join(self.scan_dir, sim_name)
        self.api.mkdir(self.sim_path)
        self.api.clone(self.sim_path, os.path.join(os.getcwd(),'devices',material))

        self.data = oghma_data()
        self.data.load(os.path.join(self.sim_path, "sim.json"))
        return self

    def clone(self, dest, src):
        gpvdm = os.path.join(src,'sim.oghma')
        json = os.path.join(src,'sim.json')
        shutil.copyfile(gpvdm,os.path.join(dest,'sim.oghma'))
        shutil.copyfile(json,os.path.join(dest,'sim.json'))
        return self

    def create_job_json(self, sim_name,material):
        self.sim_path = os.path.join(self.scan_dir, sim_name)
        try:
            os.mkdir(self.sim_path)
        except:
            "File Exists"

        self.clone(self.sim_path, os.path.join(os.getcwd(),'devices',material))
        #material = material + 'N'

        self.data = oghma_api()
        #self.data.load(os.path.join(self.sim_path, "sim.json"))
        return self

    def load_job(self, sim_name):
        self.sim_path = os.path.join(self.scan_dir, sim_name)
        #self.api.mkdir(self.sim_path)
        #self.api.clone(self.sim_path, os.getcwd())
        self.data = oghma_data()
        self.data.load(os.path.join(self.sim_path, "sim.json"))
        return self

    def modify_parameter(self, category, layer_name, layer_value, subcategory, parameter, value):
        x = getattr(self.data, category)
        x = getattr(x, layer_name)
        x = getattr(x[layer_value], subcategory)
        setattr(x, parameter, value)
        return self

    def modify_pm(self, *args, category, layer_name=None, layer_number=None, value):
        x = getattr(self.data, category[0])
        try:
            for i in category[1:]:
                x = getattr(x, i)
        except:
            print("")
        try:
            x = getattr(x, layer_name)
        except:
            print("")
        if layer_number == None:
            x = x
        else:
            x = x[layer_number]
        for arg in args[:-1]:
            x = getattr(x, arg)
        setattr(x, args[-1], value)

    def modify_pm_json(self, *args, category, layer_name=None, layer_number=None, value):
        with open(os.path.join(self.sim_path,'sim.json')) as f:
            lines = f.readlines()
            lines = "".join(lines)
            data = json.loads(lines)

        if len(category) < 2:
            try:
                data[category[0]][layer_name + str(layer_number)][args[0]] = value
            except:
                try:
                    data[category[0]][args[0]] = value
                except:
                    print(category)
                    print("Error")
        else:
            if category[0] == 'epitaxy':
                data[category[0]][layer_name + str(layer_number)][category[1]][args[0]] = value
            elif category[0] == 'mesh':
                data[category[0]][category[1]][layer_name + str(layer_number)][args[0]] = value
            elif category[0] == 'optical':
                data[category[0]][category[1]][category[2]][layer_name + str(layer_number)][args[0]][args[1]][layer_name + str(layer_number)][args[2]] = value

        jstr = json.dumps(data, sort_keys=False)
        with open(os.path.join(self.sim_path, 'sim.json'),'w') as f:
            f.write(jstr)
            f.close()

    def modify_temperature(self, temperature):
        x = getattr(self.data, 'epitaxy')
        layers = getattr(x, 'layers')
        for layer in layers:
            DOS = getattr(layer, 'shape_dos')
            setattr(DOS, 'Tstart', float(temperature))
            setattr(DOS, 'Tstop', float(temperature + 5))
        return self
    
    def modify_irradiance(self, irradiance):
        x = getattr(self.data, 'light')
        setattr(x, 'Psun', float(irradiance))
        return self

    def remesh(self):

        self.data.mesh.config.remesh_x = "True"
        self.data.mesh.config.remesh_y = "True"
        self.data.mesh.config.remesh_z = "True"

    def save_job(self):
        #self.data.save()
        self.api.add_job(path=self.sim_path)
        return self

    def run_json(self,name,n):
        ogcwd = os.getcwd()
        dirs = [os.path.join(ogcwd,'GPVDM',"Temp" + name + str(i))for i in range(n)]
        for dir in dirs:
            os.chdir(dir)
            os.system('oghma_core')
        os.chdir(ogcwd)
        return self

    def run(self):
        #self.api.server.simple_run()
        try:
            func_timeout.func_timeout(600,self.api.server.simple_run)
        except:
            print("OGHMA TIMED OUT!")

        return self
