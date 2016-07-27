# BLASX
It's a heterogeneous multiGPU level-3 BLAS library. This is an alternative to cuBLAS-XT, a commercial licensed multiGPU tiled BLAS. However, BLASX deliveries at least 25% more performance and 200% less communication volume. For detailed information, please refer to our ICS paper @ http://arxiv.org/abs/1510.05041 or http://dl.acm.org/citation.cfm?id=2926256

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

@inproceedings{wang2016blasx,
  title={BLASX: A High Performance Level-3 BLAS Library for Heterogeneous Multi-GPU Computing},
  author={Wang, Linnan and Wu, Wei and Xu, Zenglin and Xiao, Jianxiong and Yang, Yi},
  booktitle={Proceedings of the 2016 International Conference on Supercomputing},
  pages={20},
  year={2016},
  organization={ACM}
}

Linnan

---ZGEMM routine courtesy to Jan Winkelmann and Paul Springer 
