import sys, os
import argparse
import subprocess

working_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),"../..")
    
def display(fname):
    """Display PDF or HTML documents.""" 
    fpath = os.path.abspath(os.path.join(working_dir,fname))
    print(f"\nDisplaying document: {fpath}\n")
    if sys.platform == "linux":
        subprocess.call(["xdg-open "+fpath],shell=True)
    elif sys.platform == "darwin":
        subprocess.call(["open "+fpath],shell=True)
    else:
        os.startfile(fpath)
        
def showManual(fname = os.path.abspath(os.path.join(working_dir,"supplements/docs/UserGuide.pdf"))):
    display(fname)
    
def showPaper(fname = os.path.abspath(os.path.join(working_dir,"supplements/docs/DSGEModelingWithPython.pdf"))):
    display(fname)
    
def showAlgorithms(fname = os.path.abspath(os.path.join(working_dir,"supplements/docs/NumericalAlgorithms.pdf"))):
    display(fname)
    
def showAPI(fname = os.path.abspath(os.path.join(working_dir,"supplements/api_docs/_build/html/index.html"))):
    display(fname)
    
def show(argv):
    """ This is a main function."""  
    parser = argparse.ArgumentParser(description="""Displays documents for options:
                                     1 - User manual,
                                     2 - Paper,
                                     3 - Numerical algorithms,
                                     4 - API documentation""")
    parser.add_argument('-doc')        
    args = parser.parse_args()
    doc = args.doc
    
    if doc == "1":
        fname = "supplements/docs/UserGuide.pdf"
        display(fname)
    elif doc == "2":
        fname = "supplements/docs/DSGEModelingWithPython.pdf"
        display(fname)
    elif doc == "3":  
        fname = "supplements/docs/NumericalAlgorithms.pdf"
        display(fname)
    elif doc == "4":  
        fname = "supplements/api_docs/_build/html/index.html" 
        display(fname)
    else:
        fname = "supplements/docs/UserGuide.pdf"
        display(fname)
        

if __name__ == "__main__":
    show(sys.argv[1:])
