PROGNAME := mpi_omp_affinity

main:
	mpicc -O2 -Wall -Wextra -fopenmp mpi_omp_affinity.c -lhwloc -o mpi_omp_affinity

clean:
	rm -f $(PROGNAME) *.o
