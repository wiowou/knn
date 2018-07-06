CCnve = nvcc
OPnve = -g -G -DMYDEBUG


all : knn 

knn : ../debug/knn
../debug/knn : cuda/kernel.cuh cuda/kernel.cu main.cpp 
	${CCnve} ${OPnve} -Icuda      cuda/kernel.cu main.cpp -o ../debug/knn


clean : 
	rm -f ../debug/knn

cleanall : clean
	rm -rf ob
	mkdir ob

