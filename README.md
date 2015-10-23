# BLASX 
It's a heterogeneous multiGPU level-3 BLAS library. This is an alternative to cuBLAS-XT, a commercial licensed multiGPU tiled BLAS. However, BLASX deliveries at least 25% more performance and 200% less communication volume. For detailed information, please refer to our paper @ http://arxiv.org/abs/1510.05041

For installation, please change the make.inc. Basically, the following varaiables need to be updated:
<ul>
<li>
1. LIBCPUBLAS: please change it to the location of CPU BLAS. OpenBLAS is highly recommended. If you're using Linux, the library extension should be '.so'. I give an OSX verison in the repository make.inc, which follows the dylib extension.
</li>
<li>
2. LIBGPUBLAS: please change it to the location of CUDA on your machine
</li>
</ul>

Once you have configured these two variables, then you're good to go. Simply Make and the library is built in the lib folder.

To use BLASX:
<ul>
<li>
1. export LD_LIBRARY_PATH to the location of BLASX lib.
</li>
<li>
2. you need to link cuBLAS when use BLASX. There is an example of linkage in the testing folder.
</li>
</ul>
Integrating BLASX with MATLAB is pretty easy,

<ul>
</li>
<li>
1. open a command line, export BLAS_VERSION=/path/to/BLASX
</li>
<li>
2. init MATLAB from command line, say typing matlab
</li>
<li>
3. that's it, enjoy!!
</li>
</ul>
For more questions, please open an issue and I will update accordingly. Enjoy!

Please cite our paper@

@article{wang2015blasx, <br>
  title={BLASX: A High Performance Level-3 BLAS Library for Heterogeneous Multi-GPU Computing}, <br>
  author={Wang, Linnan and Wu, Wei and Xiao, Jianxiong and Yang, Yi}, <br>
  journal={arXiv preprint arXiv:1510.05041},<br>
  year={2015}<br>
}

Linnan
