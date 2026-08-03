#define _GNU_SOURCE

#include <hwloc.h>
#include <limits.h>
#include <mpi.h>
#include <omp.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define DEFAULT_TEST_MEMORY_PER_RANK_MB 64
#define OUTPUT_LINE_SIZE 4096

typedef struct {
    long mem_per_cpu_mb;
    long mem_per_node_mb;
    long mem_per_gpu_mb;
    long cpus_per_task;
    long number_of_tasks;

    long estimated_mem_per_task_mb;
    long estimated_total_job_mem_mb;

    const char *memory_request_type;
} slurm_memory_info_t;

static long parse_environment_long(const char *name)
{
    const char *value = getenv(name);
    char *end = NULL;
    long parsed;

    if (value == NULL || *value == '\0') {
        return -1;
    }

    parsed = strtol(value, &end, 10);

    if (end == value || *end != '\0' || parsed < 0) {
        return -1;
    }

    return parsed;
}

static void get_slurm_memory_info(slurm_memory_info_t *info)
{
    memset(info, 0, sizeof(*info));

    info->mem_per_cpu_mb = parse_environment_long("SLURM_MEM_PER_CPU");
    info->mem_per_node_mb = parse_environment_long("SLURM_MEM_PER_NODE");
    info->mem_per_gpu_mb = parse_environment_long("SLURM_MEM_PER_GPU");
    info->cpus_per_task = parse_environment_long("SLURM_CPUS_PER_TASK");
    info->number_of_tasks = parse_environment_long("SLURM_NTASKS");

    info->estimated_mem_per_task_mb = -1;
    info->estimated_total_job_mem_mb = -1;
    info->memory_request_type = "not detected";

    /*
     * --mem-per-cpu
     *
     * Approximate memory associated with one MPI task:
     *
     *     memory per CPU × CPUs per task
     *
     * Slurm may enforce the resulting memory as a shared job or step cgroup
     * limit rather than as an independent limit for each MPI process.
     */
    if (info->mem_per_cpu_mb >= 0) {
        info->memory_request_type = "--mem-per-cpu";

        if (info->cpus_per_task > 0) {
            info->estimated_mem_per_task_mb =
                info->mem_per_cpu_mb * info->cpus_per_task;
        }

        if (info->estimated_mem_per_task_mb >= 0 &&
            info->number_of_tasks > 0) {
            info->estimated_total_job_mem_mb =
                info->estimated_mem_per_task_mb *
                info->number_of_tasks;
        }

        return;
    }

    /*
     * --mem
     *
     * SLURM_MEM_PER_NODE is the requested memory per allocated node.
     * A meaningful per-task value cannot generally be inferred when tasks
     * may be distributed unevenly across multiple nodes.
     */
    if (info->mem_per_node_mb >= 0) {
        info->memory_request_type = "--mem / memory per node";
        return;
    }

    if (info->mem_per_gpu_mb >= 0) {
        info->memory_request_type = "--mem-per-gpu";
    }
}

static void print_optional_memory_value(
    FILE *stream,
    const char *label,
    long value,
    const char *unit)
{
    if (value >= 0) {
        fprintf(stream, "%-34s %ld %s\n", label, value, unit);
    } else {
        fprintf(stream, "%-34s not available\n", label);
    }
}

static void format_hwloc_index(
    unsigned index,
    char *buffer,
    size_t buffer_size)
{
    if (index == HWLOC_UNKNOWN_INDEX) {
        snprintf(buffer, buffer_size, "unknown");
    } else {
        snprintf(buffer, buffer_size, "%u", index);
    }
}

static const char *memory_policy_name(hwloc_membind_policy_t policy)
{
    switch (policy) {
        case HWLOC_MEMBIND_DEFAULT:
            return "default";

        case HWLOC_MEMBIND_FIRSTTOUCH:
            return "first-touch";

        case HWLOC_MEMBIND_BIND:
            return "bind";

        case HWLOC_MEMBIND_INTERLEAVE:
            return "interleave";

#ifdef HWLOC_MEMBIND_WEIGHTED_INTERLEAVE
        case HWLOC_MEMBIND_WEIGHTED_INTERLEAVE:
            return "weighted-interleave";
#endif

        case HWLOC_MEMBIND_NEXTTOUCH:
            return "next-touch";

        case HWLOC_MEMBIND_MIXED:
            return "mixed";

        default:
            return "unknown";
    }
}

static void numa_nodes_for_cpuset(
    hwloc_topology_t topology,
    hwloc_const_cpuset_t cpuset,
    char *buffer,
    size_t buffer_size)
{
    int numa_count;
    int first = 1;
    size_t used = 0;

    buffer[0] = '\0';

    if (cpuset == NULL || hwloc_bitmap_iszero(cpuset)) {
        snprintf(buffer, buffer_size, "unknown");
        return;
    }

    numa_count = hwloc_get_nbobjs_by_type(
        topology,
        HWLOC_OBJ_NUMANODE);

    for (int i = 0; i < numa_count; i++) {
        hwloc_obj_t numa_node =
            hwloc_get_obj_by_type(topology, HWLOC_OBJ_NUMANODE, i);

        unsigned index;
        int written;

        if (numa_node == NULL || numa_node->cpuset == NULL) {
            continue;
        }

        if (!hwloc_bitmap_intersects(cpuset, numa_node->cpuset)) {
            continue;
        }

        index = numa_node->os_index;

        if (index == HWLOC_UNKNOWN_INDEX) {
            index = numa_node->logical_index;
        }

        written = snprintf(
            buffer + used,
            buffer_size - used,
            "%s%u",
            first ? "" : ",",
            index);

        if (written < 0 ||
            (size_t)written >= buffer_size - used) {
            break;
        }

        used += (size_t)written;
        first = 0;
    }

    if (first) {
        snprintf(buffer, buffer_size, "unknown");
    }
}

static int parse_test_memory_size_mb(int argc, char **argv)
{
    char *end = NULL;
    long value;

    if (argc < 2) {
        return DEFAULT_TEST_MEMORY_PER_RANK_MB;
    }

    value = strtol(argv[1], &end, 10);

    if (end == argv[1] ||
        *end != '\0' ||
        value <= 0 ||
        value > 1024L * 1024L) {
        fprintf(
            stderr,
            "Invalid test memory size '%s'. "
            "Expected a positive number of MiB.\n",
            argv[1]);

        return -1;
    }

    return (int)value;
}

int main(int argc, char **argv)
{
    int mpi_rank;
    int mpi_size;
    int provided_thread_level;
    int hostname_length;

    char hostname[MPI_MAX_PROCESSOR_NAME];

    hwloc_topology_t topology;

    int test_memory_per_rank_mb;
    size_t page_size;
    size_t test_memory_size;
    size_t total_pages;

    unsigned char *test_memory = NULL;
    char *thread_output = NULL;

    int maximum_threads;
    int actual_thread_count = 1;

    slurm_memory_info_t slurm_memory;

    test_memory_per_rank_mb =
        parse_test_memory_size_mb(argc, argv);

    if (test_memory_per_rank_mb < 1) {
        return EXIT_FAILURE;
    }

    if (MPI_Init_thread(
            &argc,
            &argv,
            MPI_THREAD_FUNNELED,
            &provided_thread_level) != MPI_SUCCESS) {
        fprintf(stderr, "MPI_Init_thread failed\n");
        return EXIT_FAILURE;
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    MPI_Get_processor_name(hostname, &hostname_length);

    get_slurm_memory_info(&slurm_memory);

    /*
     * MPI is the authoritative source for the number of processes in this
     * execution. Use it when SLURM_NTASKS is unavailable.
     */
    if (slurm_memory.number_of_tasks < 0) {
        slurm_memory.number_of_tasks = mpi_size;
    }

    if (slurm_memory.mem_per_cpu_mb >= 0 &&
        slurm_memory.cpus_per_task > 0) {
        slurm_memory.estimated_mem_per_task_mb =
            slurm_memory.mem_per_cpu_mb *
            slurm_memory.cpus_per_task;

        slurm_memory.estimated_total_job_mem_mb =
            slurm_memory.estimated_mem_per_task_mb *
            slurm_memory.number_of_tasks;
    }

    if (provided_thread_level < MPI_THREAD_FUNNELED) {
        if (mpi_rank == 0) {
            fprintf(
                stderr,
                "The MPI implementation did not provide "
                "MPI_THREAD_FUNNELED support.\n");
        }

        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    if (hwloc_topology_init(&topology) != 0) {
        fprintf(
            stderr,
            "MPI rank %d: hwloc_topology_init failed\n",
            mpi_rank);

        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    if (hwloc_topology_load(topology) != 0) {
        fprintf(
            stderr,
            "MPI rank %d: hwloc_topology_load failed\n",
            mpi_rank);

        hwloc_topology_destroy(topology);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    page_size = (size_t)sysconf(_SC_PAGESIZE);

    if (page_size == 0) {
        fprintf(
            stderr,
            "MPI rank %d: unable to determine page size\n",
            mpi_rank);

        hwloc_topology_destroy(topology);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    test_memory_size =
        (size_t)test_memory_per_rank_mb * 1024UL * 1024UL;

    test_memory_size =
        ((test_memory_size + page_size - 1) / page_size) *
        page_size;

    total_pages = test_memory_size / page_size;

    if (posix_memalign(
            (void **)&test_memory,
            page_size,
            test_memory_size) != 0) {
        fprintf(
            stderr,
            "MPI rank %d: unable to allocate %d MiB\n",
            mpi_rank,
            test_memory_per_rank_mb);

        hwloc_topology_destroy(topology);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    maximum_threads = omp_get_max_threads();

    thread_output = calloc(
        (size_t)maximum_threads,
        OUTPUT_LINE_SIZE);

    if (thread_output == NULL) {
        fprintf(
            stderr,
            "MPI rank %d: unable to allocate output buffer\n",
            mpi_rank);

        free(test_memory);
        hwloc_topology_destroy(topology);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

#pragma omp parallel \
    shared(actual_thread_count, test_memory, thread_output)
    {
        int omp_thread_id = omp_get_thread_num();
        int omp_thread_count = omp_get_num_threads();

        int current_linux_cpu;

        size_t first_page;
        size_t last_page;
        size_t thread_page_count;

        unsigned char *thread_memory;
        size_t thread_memory_size;

        hwloc_obj_t pu = NULL;
        hwloc_obj_t core = NULL;
        hwloc_obj_t package = NULL;

        hwloc_bitmap_t cpu_affinity = hwloc_bitmap_alloc();
        hwloc_bitmap_t current_cpu_set = hwloc_bitmap_alloc();
        hwloc_bitmap_t memory_policy_nodes = hwloc_bitmap_alloc();
        hwloc_bitmap_t memory_location_nodes = hwloc_bitmap_alloc();

        hwloc_membind_policy_t memory_policy =
            HWLOC_MEMBIND_DEFAULT;

        char *cpu_affinity_text = NULL;
        char *memory_policy_nodes_text = NULL;
        char *memory_location_text = NULL;

        char current_cpu_numa_text[128];

        char pu_os_text[32];
        char core_os_text[32];
        char package_os_text[32];

        unsigned pu_logical_index = HWLOC_UNKNOWN_INDEX;
        unsigned pu_os_index = HWLOC_UNKNOWN_INDEX;

        unsigned core_logical_index = HWLOC_UNKNOWN_INDEX;
        unsigned core_os_index = HWLOC_UNKNOWN_INDEX;

        unsigned package_logical_index = HWLOC_UNKNOWN_INDEX;
        unsigned package_os_index = HWLOC_UNKNOWN_INDEX;

#pragma omp single
        {
            actual_thread_count = omp_thread_count;
        }

        first_page =
            total_pages * (size_t)omp_thread_id /
            (size_t)omp_thread_count;

        last_page =
            total_pages * (size_t)(omp_thread_id + 1) /
            (size_t)omp_thread_count;

        thread_page_count = last_page - first_page;

        thread_memory =
            test_memory + first_page * page_size;

        thread_memory_size =
            thread_page_count * page_size;

        /*
         * First-touch each thread's distinct page range.
         */
        for (size_t page = first_page;
             page < last_page;
             page++) {
            test_memory[page * page_size] =
                (unsigned char)(mpi_rank + omp_thread_id);
        }

#pragma omp barrier

        current_linux_cpu = sched_getcpu();

        if (current_linux_cpu >= 0) {
            hwloc_bitmap_only(
                current_cpu_set,
                (unsigned)current_linux_cpu);

            pu = hwloc_get_pu_obj_by_os_index(
                topology,
                (unsigned)current_linux_cpu);
        }

        if (pu != NULL) {
            pu_logical_index = pu->logical_index;
            pu_os_index = pu->os_index;

            core = hwloc_get_ancestor_obj_by_type(
                topology,
                HWLOC_OBJ_CORE,
                pu);

            package = hwloc_get_ancestor_obj_by_type(
                topology,
                HWLOC_OBJ_PACKAGE,
                pu);
        }

        if (core != NULL) {
            core_logical_index = core->logical_index;
            core_os_index = core->os_index;
        }

        if (package != NULL) {
            package_logical_index = package->logical_index;
            package_os_index = package->os_index;
        }

        format_hwloc_index(
            pu_os_index,
            pu_os_text,
            sizeof(pu_os_text));

        format_hwloc_index(
            core_os_index,
            core_os_text,
            sizeof(core_os_text));

        format_hwloc_index(
            package_os_index,
            package_os_text,
            sizeof(package_os_text));

        if (cpu_affinity != NULL &&
            hwloc_get_cpubind(
                topology,
                cpu_affinity,
                HWLOC_CPUBIND_THREAD) == 0) {
            hwloc_bitmap_list_asprintf(
                &cpu_affinity_text,
                cpu_affinity);
        }

        if (memory_policy_nodes != NULL &&
            hwloc_get_membind(
                topology,
                memory_policy_nodes,
                &memory_policy,
                HWLOC_MEMBIND_THREAD |
                HWLOC_MEMBIND_BYNODESET) == 0) {
            hwloc_bitmap_list_asprintf(
                &memory_policy_nodes_text,
                memory_policy_nodes);
        }

        if (thread_memory_size > 0 &&
            memory_location_nodes != NULL &&
            hwloc_get_area_memlocation(
                topology,
                thread_memory,
                thread_memory_size,
                memory_location_nodes,
                HWLOC_MEMBIND_BYNODESET) == 0) {
            hwloc_bitmap_list_asprintf(
                &memory_location_text,
                memory_location_nodes);
        }

        numa_nodes_for_cpuset(
            topology,
            current_cpu_set,
            current_cpu_numa_text,
            sizeof(current_cpu_numa_text));

        snprintf(
            thread_output +
                (size_t)omp_thread_id * OUTPUT_LINE_SIZE,
            OUTPUT_LINE_SIZE,

            "  MPI rank ID %d (rank %d of %d)\n"
            "    OpenMP thread ID %d (thread %d of %d)\n"
            "    CPU placement:\n"
            "      Linux CPU ID:             %d\n"
            "      hwloc PU logical index:   %u\n"
            "      hwloc PU OS index:        %s\n"
            "      Core logical index:       %u\n"
            "      Core OS index:            %s\n"
            "      Socket logical index:     %u\n"
            "      Socket OS index:          %s\n"
            "      NUMA node OS index(es):   %s\n"
            "      CPU affinity mask:        {%s}\n"
            "    Memory placement:\n"
            "      Memory policy:            %s\n"
            "      Policy NUMA nodes:        {%s}\n"
            "      Test memory touched:      %zu MiB\n"
            "      Physical NUMA location:   {%s}\n",

            mpi_rank,
            mpi_rank + 1,
            mpi_size,

            omp_thread_id,
            omp_thread_id + 1,
            omp_thread_count,

            current_linux_cpu,

            pu_logical_index,
            pu_os_text,

            core_logical_index,
            core_os_text,

            package_logical_index,
            package_os_text,

            current_cpu_numa_text,

            cpu_affinity_text != NULL
                ? cpu_affinity_text
                : "unavailable",

            memory_policy_name(memory_policy),

            memory_policy_nodes_text != NULL
                ? memory_policy_nodes_text
                : "unavailable",

            thread_memory_size / 1024UL / 1024UL,

            memory_location_text != NULL
                ? memory_location_text
                : "unavailable");

        free(cpu_affinity_text);
        free(memory_policy_nodes_text);
        free(memory_location_text);

        hwloc_bitmap_free(cpu_affinity);
        hwloc_bitmap_free(current_cpu_set);
        hwloc_bitmap_free(memory_policy_nodes);
        hwloc_bitmap_free(memory_location_nodes);
    }

    /*
     * Build one complete report per rank in memory. Only rank 0 writes to
     * stdout, avoiding interleaving by Slurm's distributed stdout collector.
     */
    char *rank_report = NULL;
    size_t rank_report_size = 0;
    FILE *rank_stream = open_memstream(&rank_report, &rank_report_size);

    if (rank_stream == NULL) {
        fprintf(stderr, "MPI rank %d: open_memstream failed\n", mpi_rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    fprintf(
        rank_stream,
        "\n"
        "============================================================\n"
        "Host:                             %s\n"
        "MPI rank ID:                      %d\n"
        "MPI rank ordinal:                 %d of %d\n"
        "OpenMP threads in rank:           %d\n"
        "Slurm memory request type:        %s\n",
        hostname,
        mpi_rank,
        mpi_rank + 1,
        mpi_size,
        actual_thread_count,
        slurm_memory.memory_request_type);

    if (slurm_memory.mem_per_cpu_mb >= 0) {
        print_optional_memory_value(
            rank_stream,
            "Slurm memory per allocated CPU:",
            slurm_memory.mem_per_cpu_mb,
            "MiB");

        print_optional_memory_value(
            rank_stream,
            "Slurm CPUs per task:",
            slurm_memory.cpus_per_task,
            "");

        print_optional_memory_value(
            rank_stream,
            "Estimated Slurm memory per task:",
            slurm_memory.estimated_mem_per_task_mb,
            "MiB");

        print_optional_memory_value(
            rank_stream,
            "Estimated total job memory:",
            slurm_memory.estimated_total_job_mem_mb,
            "MiB");
    } else if (slurm_memory.mem_per_node_mb >= 0) {
        print_optional_memory_value(
            rank_stream,
            "Slurm memory per node:",
            slurm_memory.mem_per_node_mb,
            "MiB");

        fprintf(
            rank_stream,
            "%-34s cannot be inferred reliably\n",
            "Estimated memory per task:");
    } else if (slurm_memory.mem_per_gpu_mb >= 0) {
        print_optional_memory_value(
            rank_stream,
            "Slurm memory per allocated GPU:",
            slurm_memory.mem_per_gpu_mb,
            "MiB");
    } else {
        fprintf(
            rank_stream,
            "%-34s no Slurm memory variable found\n",
            "Slurm memory details:");
    }

    fprintf(
        rank_stream,
        "%-34s %d MiB\n"
        "%-34s %zu MiB\n"
        "============================================================\n",
        "Test allocation per MPI rank:",
        test_memory_per_rank_mb,
        "Total test allocation in MPI job:",
        (size_t)test_memory_per_rank_mb * (size_t)mpi_size);

    for (int thread = 0; thread < actual_thread_count; thread++) {
        fputs(
            thread_output + (size_t)thread * OUTPUT_LINE_SIZE,
            rank_stream);

        if (thread + 1 < actual_thread_count) {
            fputc('\n', rank_stream);
        }
    }

    if (fclose(rank_stream) != 0) {
        fprintf(stderr, "MPI rank %d: closing report stream failed\n", mpi_rank);
        free(rank_report);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    if (rank_report_size > (size_t)INT_MAX) {
        fprintf(stderr, "MPI rank %d: report is too large for MPI_Gatherv\n", mpi_rank);
        free(rank_report);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    int local_report_size = (int)rank_report_size;
    int *report_sizes = NULL;
    int *report_displacements = NULL;
    char *all_reports = NULL;

    if (mpi_rank == 0) {
        report_sizes = calloc((size_t)mpi_size, sizeof(*report_sizes));
        report_displacements = calloc(
            (size_t)mpi_size,
            sizeof(*report_displacements));

        if (report_sizes == NULL || report_displacements == NULL) {
            fprintf(stderr, "MPI rank 0: unable to allocate gather metadata\n");
            free(report_sizes);
            free(report_displacements);
            free(rank_report);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    MPI_Gather(
        &local_report_size,
        1,
        MPI_INT,
        report_sizes,
        1,
        MPI_INT,
        0,
        MPI_COMM_WORLD);

    int total_report_size = 0;

    if (mpi_rank == 0) {
        for (int rank = 0; rank < mpi_size; rank++) {
            report_displacements[rank] = total_report_size;

            if (report_sizes[rank] > INT_MAX - total_report_size) {
                fprintf(stderr, "MPI rank 0: combined report is too large\n");
                free(report_sizes);
                free(report_displacements);
                free(rank_report);
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }

            total_report_size += report_sizes[rank];
        }

        all_reports = malloc((size_t)total_report_size);

        if (all_reports == NULL && total_report_size > 0) {
            fprintf(stderr, "MPI rank 0: unable to allocate combined report\n");
            free(report_sizes);
            free(report_displacements);
            free(rank_report);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    MPI_Gatherv(
        rank_report,
        local_report_size,
        MPI_CHAR,
        all_reports,
        report_sizes,
        report_displacements,
        MPI_CHAR,
        0,
        MPI_COMM_WORLD);

    if (mpi_rank == 0 && total_report_size > 0) {
        size_t written = fwrite(
            all_reports,
            1,
            (size_t)total_report_size,
            stdout);

        if (written != (size_t)total_report_size) {
            fprintf(stderr, "MPI rank 0: failed to write complete report\n");
        }

        fflush(stdout);
    }

    free(all_reports);
    free(report_displacements);
    free(report_sizes);
    free(rank_report);

    free(thread_output);
    free(test_memory);

    hwloc_topology_destroy(topology);

    MPI_Finalize();

    return EXIT_SUCCESS;
}
