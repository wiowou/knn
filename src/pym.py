from make import Source,Target,Makefile
from make.compiler.nvcc import Exe, Shared

import glob
import os

if __name__ == '__main__':
    targetName='knn'
    debug=True
    binType='exe'
    
    binOutputDir='../bin'
    if debug:
        binOutputDir='../debug'
    if binType == 'exe':
        bname=os.path.join(binOutputDir,targetName)
        binary=Exe()
    elif binType == 'lib':
        bname=os.path.join(binOutputDir,'lib'+targetName+'.so')
        binary=Shared()
    if debug:
        binary.options+='-g -G -DMYDEBUG'

    binary.includeDir = ['cuda']
    
    binTarg=Target(bname,binary)
    binTarg.source=[]
    cFiles=glob.glob1('cuda','*.cu') #list all .cu files in the cuda directory
    for f in cFiles:
        binTarg.source.append(Source(os.path.join('cuda',f)))
   
    if binType == 'exe':
        cFiles=glob.glob1('.','*.cpp') #list all .cpp files in the current directory
        for f in cFiles:
            binTarg.source.append(Source(f))
    
    binTarg.target=[]
    cFiles=glob.glob1('cuda','*.cuh') #list all .cuh files in the cuda directory
    for f in cFiles:
        binTarg.target.append(Source(os.path.join('cuda',f)))

    makefile=Makefile()
    makefile.target=[binTarg]
    makefile.write()
