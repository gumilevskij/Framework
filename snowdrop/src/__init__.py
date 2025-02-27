import os,sys

path = os.path.dirname(os.path.abspath(__file__))
working_dir = os.path.abspath(path + "/../..")
sys.path.append(working_dir)
os.chdir(working_dir)

__all__ = ["dignad","dignar","epidemic","graphs","gui","info",
            "misc","model","numeric","preprocessor","samples",
            "tests","utils","driver","info"]


