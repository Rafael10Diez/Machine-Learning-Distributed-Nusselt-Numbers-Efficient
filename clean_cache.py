# ------------------------------------------------------------------------
#                  Generic Libraries
# ------------------------------------------------------------------------
from   os      import  walk
from   os.path import  dirname, basename, abspath
from   shutil  import  rmtree

# ------------------------------------------------------------------------
#                  Main Function
# ------------------------------------------------------------------------

def clean_pycache(folder):
    for x in walk(folder):
        x = x[0]
        if basename(x)  ==  '__pycache__':
            try:
                rmtree(x)
            except:
                print(f'WARNING: failed to delete (pycache = {x})')
    print(f'\nCleaned all __pycache__ (root = {folder})\n')

# ------------------------------------------------------------------------
#                  Direct Runner
# ------------------------------------------------------------------------

if __name__ == '__main__':
    clean_pycache(dirname(abspath(__file__)))
