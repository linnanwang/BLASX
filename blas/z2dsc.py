#!/usr/bin/python

import os, sys

BLAS = [
        'gemm',
        'trsm',
        'trmm',
        'syrk',
        'syr2k',
        'symm',
        ]

CZBLAS = [
          'herk',
          'her2k',
          'hemm',
         ]

def clean(fname):
    if fname in BLAS:
        cmd = "rm -f s" + fname + ".c"
        print cmd
        os.system(cmd)
        cmd = "rm -f d" + fname + ".c"
        print cmd 
        os.system(cmd)
        cmd = "rm -f c" + fname + ".c"
        print cmd 
        os.system(cmd)
    if fname in CZBLAS:
        cmd = "rm -f c" + fname + ".c"
        print cmd
        os.system(cmd)

def generate(fname):
    if fname in BLAS:
        cmd = "python tools/codegen.py -p s -f z" + fname + ".c"
        print cmd
        os.system(cmd)
        cmd = "python tools/codegen.py -p d -f z" + fname + ".c"
        print cmd
        os.system(cmd)
        cmd = "python tools/codegen.py -p c -f z" + fname + ".c"
        print cmd
        os.system(cmd)
    if fname in CZBLAS:
        cmd = "python tools/codegen.py -p c -f z" + fname + ".c"
        print cmd
        os.system(cmd)

def fix_ds_pointer(fname):
    cmd = "python ds_pointer_fix.py " + fname
    print cmd
    os.system(cmd)

def cleanall():
    for i in range(0, len(BLAS)):
        clean(BLAS[i])
    for i in range(0, len(CZBLAS)):
            clean(CZBLAS[i])

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == 'cleanall':
            cleanall();
        else:
            fname = sys.argv[1]
            if (fname in BLAS) or (fname in CZBLAS):
                clean(fname)
                generate(fname)
                fix_ds_pointer(fname)
            elif (fname[1:] in BLAS) or (fname[1:] in CZBLAS):
                clean(fname[1:])
                generate(fname[1:])
                fix_ds_pointer(fname[1:])
            else:
                print "no such file, skip"
                exit(0)
    else:
        for i in range(0, len(BLAS)):
            clean(BLAS[i])
            generate(BLAS[i])
            fix_ds_pointer(BLAS[i])
        for i in range(0, len(CZBLAS)):
            clean(CZBLAS[i])
            generate(CZBLAS[i])
