PROGNAME := mpi_omp_affinity
CC       := mpicc
CFLAGS   := -O2 -Wall -Wextra -fopenmp
LDLIBS   := -lhwloc

.PHONY: all clean info

all: $(PROGNAME)

$(PROGNAME): mpi_omp_affinity.c
	$(CC) $(CFLAGS) $< $(LDLIBS) -o $@

info:
	@echo "Compiler wrapper: $$(command -v $(CC))"
	@$(CC) --version | head -n 1
	@echo "hwloc: $$(command -v hwloc-info)"

clean:
	rm -f $(PROGNAME) *.o
