all: hopping 

#geometry: test_geometry.c ../lib/start.c ../lib/dirac.c ../lib/utils.c   ../opqcd-random/ranlux.c  ../opqcd-random/ranlxd.c  ../opqcd-random/ranlxs.c ../opqcd-random/gauss.c
#	cc  -O3 -o geometry test_geometry.c ../lib/start.c ../lib/dirac.c ../lib/utils.c ../opqcd-random/ranlux.c  ../opqcd-random/ranlxd.c  ../opqcd-random/ranlxs.c  ../opqcd-random/gauss.c
hopping: test_hopping.c ../lib/start.c ../lib/dirac.c ../lib/utils.c 
	#mpicc  -std=gnu99  -fopenmp   -O3 -Wall   -Wextra   -DSSE3 -msse3 -DSSE_INTRIN -DP4 -o hopping test_hopping.c ../lib/start.c ../lib/dirac.c ../lib/utils.c ../lib/su3_256.c -I ../include -lm
	mpiicc  -std=gnu99 -fopenmp -O3 -Wall -mmic  -Wextra  -DMIC -DTEST_HOPPING_MIC -DMIC_PREFETCH  -o hopping test_hopping.c ../lib/start.c ../lib/dirac.c ../lib/utils.c  -I ../include -lm
	#mpiicc  -std=gnu99 -fopenmp  -O3 -Wall -mmic  -Wextra  -DMIC -DTEST_SU3MUL_MIC -o hopping test_hopping.c ../lib/start.c ../lib/dirac.c ../lib/utils.c  -I ../include -lm
	#mpiicc  -std=gnu99 -openmp  -openmp -opt-prefetch-distance=64,8 -opt-streaming-cache-evict=0 -opt-streaming-stores always -O3 -Wall   -Wextra  -mmic -DP4 -DMPI -DMIC -o hopping test_hopping.c ../lib/start.c ../lib/dirac.c ../lib/utils.c  -I ../include -lm
	#mpicc  -std=gnu99  -fopenmp   -O3 -Wall   -Wextra   -DSSE3 -msse3 -DAVX -mavx -DP4 -o hopping test_hopping.c ../lib/start.c ../lib/dirac.c ../lib/utils.c ../lib/su3_256.c -I ../include -lm
	#mpicc  -std=gnu99  -DSSE2  -g -O0 -Wall -msse2 -DPENTIUM -Wextra  -o hopping test_hopping.c ../lib/start.c ../lib/dirac.c ../lib/utils.c -I ../include -lm
#fpc_hopping: fpc_hopping.c ../lib/start.c ../lib/dirac_fpc.c ../lib/utils.c  ../opqcd-random/ranlux.c  ../opqcd-random/ranlxd.c  ../opqcd-random/ranlxs.c ../opqcd-random/gauss.c ../lib/fpc.c
#	cc  -O3 -o fpc_hopping fpc_hopping.c ../lib/start.c ../lib/dirac_fpc.c ../lib/utils.c ../opqcd-random/ranlux.c  ../opqcd-random/ranlxd.c  ../opqcd-random/ranlxs.c ../opqcd-random/gauss.c ../lib/fpc.c
#lzo_hopping: lzo_hopping.c ../lib/start.c ../lib/dirac_lzo.c ../lib/utils.c ../lib/minilzo.c  ../opqcd-random/ranlux.c  ../opqcd-random/ranlxd.c  ../opqcd-random/ranlxs.c ../opqcd-random/gauss.c
#	cc  -O3 -o lzo_hopping lzo_hopping.c ../lib/start.c ../lib/dirac_lzo.c ../lib/utils.c ../lib/minilzo.c  ../opqcd-random/ranlux.c  ../opqcd-random/ranlxd.c  ../opqcd-random/ranlxs.c ../opqcd-random/gauss.c
clean:
	rm hopping
