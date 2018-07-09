CCnvs = nvcc
OPnvs = --compiler-options '-fPIC' -g -G -DMYDEBUG


all : libknn.so 

libknn.so : ../debug/libknn.so
../debug/libknn.so : cuda/kernel.cuh cuda/kernel.cu 
	${CCnvs} --shared ${OPnvs} -Icuda cuda/kernel.cu -o ../debug/libknn.so


clean : 
	rm -f ../debug/libknn.so

cleanall : clean
	rm -rf ob
	mkdir ob

