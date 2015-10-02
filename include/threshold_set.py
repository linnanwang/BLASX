import re
import sys

argument=sys.argv

if len(sys.argv) < 3:
    print "ERROR:At least 2 input arguments, precisions(f, d, c, z) & type (blas,cublas,cublasxt)"
    exit(1)

if argument[1] == "f":
    blas = re.compile("#define S[A-Z]+[0-9]?[A-Z]+_((CU)?BLAS(XT)?)+_THRESHOLD+")
elif argument[1] == "d":
    blas = re.compile("#define D[A-Z]+[0-9]?[A-Z]+_((CU)?BLAS(XT)?)+_THRESHOLD+")
elif argument[1] == "c":
    blas = re.compile("#define C[A-Z]+[0-9]?[A-Z]+_((CU)?BLAS(XT)?)+_THRESHOLD+")
elif argument[1] == "z":
    blas = re.compile("#define Z[A-Z]+[0-9]?[A-Z]+_((CU)?BLAS(XT)?)+_THRESHOLD+")
else:
    print "ERROR:only f, d, c, z allowed"
    exit(1)

if argument[2] == "blas":
    blas_threshold = 10000
    cublas_threshold = 20000
elif argument[2] == "cublas":
    blas_threshold = -100
    cublas_threshold = 20000
elif argument[2] == "cublasxt":
    blas_threshold = -100
    cublas_threshold = -50
else:
    print "ERROR:only blas,cublas,cublasxt allowed"
    exit(1)

with open('blasx_config.h', 'r') as fin:
    data = fin.readlines()
fin.close()

for index,line in enumerate(data):
    result = blas.match(line)
    if result is not None:
        print line
        if line.find("_CUBLASXT_") > 0:
            newline = result.group(0)+" \n"
            data[index] = newline
        elif line.find("_CUBLAS_") > 0:
            newline = result.group(0)+" "+str(cublas_threshold)+"\n"
            data[index] = newline
        elif line.find("_BLAS_") > 0:
            newline = result.group(0)+" "+str(blas_threshold)+"\n"
            data[index] = newline

with open('blasx_config.h', 'w') as fout:
    fout.writelines(data)
fout.close()
