import time
import os
import pickle
from torch import save, load
import sys

GIGA = 1024**3

def timeDiffFormated(start):
    tock = time.time()
    total = tock - start
    hours = int(total//3600)
    mins = int(total%3600//60)
    secs = int(total%3600%60//1)
    if hours > 0:
        s = "{}h {}m {}s".format(hours, mins, secs)
    elif mins > 0:
        s = "{}m {}s".format(mins, secs)
    elif secs > 0:
        s = "{}s".format(secs)
    else:
        s = "< 0s"
    return s, tock

def goToDir(dir):
    home = os.getenv('HOME')
    try:
        os.chdir(os.path.join(home, dir))
    except:
        os.chdir(home)
        os.makedirs(dir)
        os.chdir(dir)
    return os.getcwd()

def timeFormated() -> str:
    return time.strftime("%Y-%b-%d_%H:%M", time.gmtime())

def createFolder(name:str, mod:str):
    start = mod + '_' + timeFormated()
    new_dir = os.path.join(name,start)
    new_dir = goToDir(new_dir)
    return start, new_dir

class Tocker:
    def __init__(self):
        self.tick
    @property
    def tick(self):
        self.start = time.time()
        return self.start
    @property
    def tock(self):
        s, self.start = timeDiffFormated(self.start)
        return s
    @property
    def tocktock(self):
        """
        Returns the time elapsed since the last tick in minutes
        """
        return (time.time() - self.start) // 60
    def lockHz(self, Hz:int):
        tHz = 1 / Hz
        remaind = time.time() - self.start
        remaind = tHz - remaind
        if remaind > 0:
            time.sleep(remaind)
            return True

class Saver:
    def __init__(self, game:str, Dir:str="DQN_results/"):
        self.start, self.dir = createFolder(Dir, game)
        self.model_saves = Stack()
        self.limitModels = 6
        self.object_saves = Stack()
        self.limitObjects = 4
        
    def saveModel(self, state_dict, name):
        name = name + "_" + timeFormated() + ".modelState"
        path_name = os.path.join(self.dir, name)
        save(state_dict, path_name)
        print("Model saved successfully in", name)
        self.model_saves.add(name)
        self.cleaner()

    def saveObject(self, obj, name, ext='pObj'):
        if obj is None:
            return None
        name = name + "_" + timeFormated() + '.' + ext
        fileHandler = open(name,'wb')
        if sys.getsizeof(obj) < 3.8*GIGA:
            pickle.dump(obj, fileHandler)
            fileHandler.close()
            print("Object saved successfully in", name)
            self.object_saves.add(name)
        else:
            fileHandler.close()
            os.remove(name)
            print("Couldn't save file", name)
        self.cleaner()

    def loadObject(self, dir_name):
        if os.path.exists(dir_name):
            fileHandler = open(dir_name, 'rb')
            obj =  pickle.load(fileHandler)
            fileHandler.close()
            print("Object successfully loaded from ",dir_name)
            return obj
        else:
            print("The path or file do not exists.")

    def loadModel(self, dir_name, device):
        if os.path.exists(dir_name):
            model = load(dir_name, map_location=device)
            print("Model successfully loaded from ",dir_name)
            return model
        else:
            print("The path or file do not exists.")

    def cleaner(self):
        actual_files = set(os.listdir())
        while len(self.model_saves) > self.limitModels:
            # Excess of models on folder
            target = self.model_saves.pop()
            if target in actual_files:
                os.remove(target)
        while len(self.object_saves) > self.limitObjects:
            # Excess of models on folder
            target = self.object_saves.pop()
            if target in actual_files:
                os.remove(target)

class Stack:
    """
    For a FILO
    """
    def __init__(self, obj=None):
        self.stack = [obj]
    def add(self, obj):
        self.stack.append(obj)
    def pop(self):
        pop = self.stack[0]
        self.stack = self.stack[1:]
        return pop
    def __len__(self):
        return len(self.stack)