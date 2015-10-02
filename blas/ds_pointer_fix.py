#!/usr/bin/python

import sys, re, string

blas_lines = {
              'gemm': 8,
              'trmm': 8,
              'syrk': 7,
              'trsm': 7,
              'syr2k': 13,
              'symm': 8,
             }

def fix_blas(matched, line, old, new):
    newstr = string.replace(line[matched.start():matched.end()], old, new)
    newstr = string.replace(newstr, " )", ")")
    newstr = line[0:matched.start()] + newstr + line[matched.end():len(line)]
    return newstr
    

def fix_pointer(fname, precision, ptype):
    filename = precision + fname + ".c"
    print filename
    ifile = open(filename, 'r')
    src_code = ifile.readlines()
    ifile.close()
    cblas_name = "cblas_" + precision + fname;
    cublas_name = "cublas" + precision.upper() + fname
    cublasxt_name = "cublasXt" + precision.upper() + fname
    i = 0;
    while (i < len(src_code)):
        # match cblas
        match_cblas = re.search(r'\s'+re.escape(cblas_name)+r'\(', src_code[i])
        if match_cblas:
            print match_cblas.group(), i
            if i > len(src_code)/2: #cblas in f77   
                for j in range(1, blas_lines[fname]):
                    if i+j >= len(src_code):
                        break
                    match_alpha = re.search(r'\('+re.escape(ptype)+r'\s*\*\)alpha', src_code[i+j])
                    if match_alpha:
                        src_code[i+j] = fix_blas(match_alpha, src_code[i+j], "*", "")
                        src_code[i+j] = fix_blas(match_alpha, src_code[i+j], "alpha", "*alpha")
                    match_beta = re.search(r'\('+re.escape(ptype)+r'\s*\*\)beta', src_code[i+j])
                    if match_beta:
                        src_code[i+j] = fix_blas(match_beta, src_code[i+j], "*", "")
                        src_code[i+j] = fix_blas(match_beta, src_code[i+j], "beta", "*beta")
            else: #cblsas in function name
                for j in range(1, blas_lines[fname]):
                    if i+j >= len(src_code):
                        break
                    match_alpha = re.search(r'const\s+'+re.escape(ptype)+r'\s+\*\s*alpha', src_code[i+j])
                    if match_alpha:
                        src_code[i+j] = fix_blas(match_alpha, src_code[i+j], "*", "")
                    match_beta = re.search(r'const\s+'+re.escape(ptype)+r'\s+\*\s*beta', src_code[i+j])
                    if match_beta:
                        src_code[i+j] = fix_blas(match_beta, src_code[i+j], "*", "")
            i = i + blas_lines[fname]

        # match cublas
        if i >= len(src_code):
            break
        match_cublas = re.search(r'\s'+re.escape(cublas_name)+r'\(', src_code[i])
        if match_cublas:
            print match_cublas.group(), i
            for j in range(1, blas_lines[fname]):
                if i+j >= len(src_code):
                        break
                match_alpha = re.search(r'('+re.escape(ptype)+r'\s*\*|\('+re.escape(ptype)+r'\s*\*\))alpha', src_code[i+j])
                if match_alpha:
                    src_code[i+j] = fix_blas(match_alpha, src_code[i+j], "alpha", "&alpha")
                match_beta = re.search(r'('+re.escape(ptype)+r'\s*\*|\('+re.escape(ptype)+r'\s*\*\))beta', src_code[i+j])
                if match_beta:
                    src_code[i+j] = fix_blas(match_beta, src_code[i+j], "beta", "&beta")
            i = i + blas_lines[fname]

        # match cublas
        if i >= len(src_code):
            break
        match_cublasxt = re.search(r'\s'+re.escape(cublasxt_name)+r'\(', src_code[i])
        if match_cublasxt:
            print match_cublasxt.group(), i
            for j in range(1, blas_lines[fname]):
                if i+j >= len(src_code):
                        break
                match_alpha = re.search(r'('+re.escape(ptype)+r'\s*\*|\('+re.escape(ptype)+r'\s*\*\))alpha', src_code[i+j])
                if match_alpha:
                    src_code[i+j] = fix_blas(match_alpha, src_code[i+j], "alpha", "&alpha")
                match_beta = re.search(r'('+re.escape(ptype)+r'\s*\*|\('+re.escape(ptype)+r'\s*\*\))beta', src_code[i+j])
                if match_beta:
                    src_code[i+j] = fix_blas(match_beta, src_code[i+j], "beta", "&beta")
            i = i + blas_lines[fname]

        i = i+1
        
    ofile = open(filename, 'w')
    ofile.writelines(src_code)
    ofile.close()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print "wrong cmd"
        exit(0)
    else:
        fix_pointer(sys.argv[1], "s", "float")
        fix_pointer(sys.argv[1], "d", "double")


