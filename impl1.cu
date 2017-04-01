#include <vector>
#include <iostream>

#include "utils.h"
#include "cuda_error_check.cuh"
#include "initial_graph.hpp"
#include "parse_graph.hpp"

#define TRUE 1
#define FALSE 0

/* Edge process out of core with shared memory */
__global__ void edge_process_out_of_core_shared_memory(unsigned int edges_length,
                            unsigned int *src,
                            unsigned int *dest,
                            unsigned int *weight,
                            unsigned int *distance_prev,
                            unsigned int *distance_cur,
                            int *noChange,
                            int *is_distance_infinity_prev,
                            int *is_distance_infinity_cur) {
    extern __shared__ unsigned int s_data[ ];
    extern __shared__ unsigned int dest_s_data[ ];
    extern __shared__ unsigned int is_dest_valid[ ];

    // unsigned int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    // unsigned int thread_num = blockDim.x * gridDim.x;
    //
    // unsigned int iter = edges_length % thread_num == 0 ? edges_length / thread_num : edges_length / thread_num + 1;
    // unsigned int lane = thread_id % 32;
    //
    // s_data[threadIdx.x] = -1;
    // is_dest_valid[threadIdx.x] = FALSE;
    // dest_s_data[threadIdx.x] = -1;
    //
    // __syncthreads();

    unsigned int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int thread_num = blockDim.x * gridDim.x;

    unsigned int warp_id = thread_id / 32;
    unsigned int warp_num = thread_num % 32 == 0 ? thread_num / 32 : thread_num / 32 + 1;

    unsigned int load = edges_length % warp_num == 0 ? edges_length / warp_num : edges_length / warp_num + 1;
    unsigned int beg = load * warp_id;
    unsigned int end = min(edges_length, beg + load);
    unsigned int lane = thread_id % 32;
    beg += lane;

    s_data[threadIdx.x] = -1;
    is_dest_valid[threadIdx.x] = FALSE;
    dest_s_data[threadIdx.x] = -1;

    __syncthreads();

    for (unsigned int i = beg; i < end; i += 32) {

    // unsigned int i;
    // for (i = 0; i < iter; i++) {
    //   unsigned int dataid = thread_id + i * thread_num;
    //   lane = dataid % 32;
    //
    //   if (dataid >= edges_length)
    //     break;

      // unsigned int u = src[dataid];
      // unsigned int v = dest[dataid];
      // unsigned int w = weight[dataid];
      unsigned int u = src[i];
      unsigned int v = dest[i];
      unsigned int w = weight[i];

      //printf("src %u | dest %u | weight %u | dataid %u | lane %u\n", u, v, w, dataid, lane);

      dest_s_data[threadIdx.x] = v;
      is_dest_valid[threadIdx.x] = TRUE;

      if (is_distance_infinity_prev[u] == TRUE) {
        s_data[threadIdx.x] = -1;
      }
      else {
        s_data[threadIdx.x] = min(distance_cur[v], distance_prev[u] + w);
      }

      //printf("%u %u %u | s_data at %u is %u, lane %u, i %u\n", u, v, w, threadIdx.x, s_data[threadIdx.x], lane, i);

      __syncthreads();

      // segmented scan to find minimum
      if (lane >= 1 && dest_s_data[threadIdx.x] == dest_s_data[threadIdx.x-1] && is_dest_valid[threadIdx.x-1] == TRUE)
        s_data[threadIdx.x] = min(s_data[threadIdx.x], s_data[threadIdx.x-1]);
      if (lane >= 2 && dest_s_data[threadIdx.x] == dest_s_data[threadIdx.x-2] && is_dest_valid[threadIdx.x-2] == TRUE)
        s_data[threadIdx.x] = min(s_data[threadIdx.x], s_data[threadIdx.x-2]);
      if (lane >= 4 && dest_s_data[threadIdx.x] == dest_s_data[threadIdx.x-4] && is_dest_valid[threadIdx.x-4] == TRUE)
        s_data[threadIdx.x] = min(s_data[threadIdx.x], s_data[threadIdx.x-4]);
      if (lane >= 8 && dest_s_data[threadIdx.x] == dest_s_data[threadIdx.x-8] && is_dest_valid[threadIdx.x-8] == TRUE)
        s_data[threadIdx.x] = min(s_data[threadIdx.x], s_data[threadIdx.x-8]);
      if (lane >= 16 && dest_s_data[threadIdx.x] == dest_s_data[threadIdx.x-16] && is_dest_valid[threadIdx.x-16] == TRUE)
        s_data[threadIdx.x] = min(s_data[threadIdx.x], s_data[threadIdx.x-16]);


      __syncthreads();

      // i is in bounds
      if (i + 1 < edges_length) {
        //printf("inside 1\n");
        // this thread is the last thread for the segment, so it holds the min
        // this thread is the last in a block
        if (threadIdx.x == blockDim.x - 1 || dest_s_data[threadIdx.x] != dest_s_data[threadIdx.x+1] || is_dest_valid[threadIdx.x+1] == FALSE) {
          //printf("inside 4\n");
          //printf("the min for dest %u is %u\n", dest[dataid], s_data[threadIdx.x]);
          int old_distance = atomicMin(&distance_cur[v], s_data[threadIdx.x]);
          if (distance_cur[v] != -1)
            atomicMin(&is_distance_infinity_cur[v], FALSE);
          // test for a change!
          if (old_distance != distance_cur[v]) {
            //printf("there is change\n");
            atomicMin(noChange, FALSE);
          }
        }
        //printf("inside no crashes\n");
      }
      // i is the last element
      else {
        //printf("the min for dest %u is %u\n", dest[i], s_data[threadIdx.x]);
        int old_distance = atomicMin(&distance_cur[v], s_data[threadIdx.x]);
        if (distance_cur[v] != -1)
          atomicMin(&is_distance_infinity_cur[v], FALSE);
        // test for a change!
        if (old_distance != distance_cur[v]) {
          //printf("there is change\n");
          atomicMin(noChange, FALSE);
        }
      }
    }
}

/* Edge process out of core with no shared memory */
__global__ void edge_process_out_of_core_non_warp_partitioned(unsigned int edges_length,
                            unsigned int *src,
                            unsigned int *dest,
                            unsigned int *weight,
                            unsigned int *distance_prev,
                            unsigned int *distance_cur,
                            int *noChange,
                            int *is_distance_infinity_prev,
                            int *is_distance_infinity_cur) {
    unsigned int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int thread_num = blockDim.x * gridDim.x;

    unsigned int iter = edges_length % thread_num == 0 ? edges_length / thread_num : edges_length / thread_num + 1;

    unsigned int i;
    for (i = 0; i < iter; i++) {
      unsigned int dataid = thread_id + i * thread_num;

      if (dataid >= edges_length)
        break;

      unsigned int u = src[dataid];
      unsigned int v = dest[dataid];
      unsigned int w = weight[dataid];
      if (is_distance_infinity_prev[u] == TRUE) {
        continue;
      }
      //printf("%u isn't infinite distance\n", u);
      if (distance_prev[u] + w < distance_prev[v]) {
        // relax
        //printf("%u %u\n", distance_cur[v], distance_prev[u] + w);
        unsigned int old_distance = atomicMin(&distance_cur[v], distance_prev[u] + w);
        atomicMin(&is_distance_infinity_cur[v], FALSE);
        //printf("%u %u %u %d\n", old_distance, distance_cur[v], distance_prev[u] + w, is_distance_infinity[v]);
        // test for a change!
        if (old_distance != distance_cur[v]) {
          //printf("there is change\n");
          atomicMin(noChange, FALSE);
        }
      }
    }
}

/* Edge process out of core with no shared memory */
/* This has been found to be 20% slower than it's alternative */
__global__ void edge_process_out_of_core_warp_sided(unsigned int edges_length,
                            unsigned int *src,
                            unsigned int *dest,
                            unsigned int *weight,
                            unsigned int *distance_prev,
                            unsigned int *distance_cur,
                            int *noChange,
                            int *is_distance_infinity_prev,
                            int *is_distance_infinity_cur) {

    unsigned int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int thread_num = blockDim.x * gridDim.x;

    unsigned int warp_id = thread_id / 32;
    unsigned int warp_num = thread_num % 32 == 0 ? thread_num / 32 : thread_num / 32 + 1;

    unsigned int load = edges_length % warp_num == 0 ? edges_length / warp_num : edges_length / warp_num + 1;
    unsigned int beg = load * warp_id;
    unsigned int end = min(edges_length, beg + load);
    unsigned int lane = thread_id % 32;
    beg += lane;
    for (unsigned int i = beg; i < end; i += 32) {
      unsigned int u = src[i];
      unsigned int v = dest[i];
      unsigned int w = weight[i];

      if (is_distance_infinity_prev[u] == TRUE) {
        continue;
      }

      //printf("%u isn't infinite distance\n", u);
      if (distance_prev[u] + w < distance_prev[v]) {
        // relax
        //printf("%u %u\n", distance_cur[v], distance_prev[u] + w);
        unsigned int old_distance = atomicMin(&distance_cur[v], distance_prev[u] + w);
        atomicMin(&is_distance_infinity_cur[v], FALSE);
        //printf("%u %u %u %d\n", old_distance, distance_cur[v], distance_prev[u] + w);
        // test for a change!
        if (old_distance != distance_cur[v]) {
          //printf("there is change\n");
          atomicMin(noChange, FALSE);
        }
      }
    }
}

/* Edge process in core */
__global__ void edge_process_in_core(unsigned int edges_length,
                            unsigned int vertices_length,
                            unsigned int *src,
                            unsigned int *dest,
                            unsigned int *weight,
                            unsigned int *distance,
                            int *noChange,
                            int *is_distance_infinity) {
    unsigned int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int thread_num = blockDim.x * gridDim.x;

    unsigned int warp_id = thread_id / 32;
    unsigned int warp_num = thread_num % 32 == 0 ? thread_num / 32 : thread_num / 32 + 1;

    unsigned int load = edges_length % warp_num == 0 ? edges_length / warp_num : edges_length / warp_num + 1;
    unsigned int beg = load * warp_id;
    unsigned int end = min(edges_length, beg + load);
    unsigned int lane = thread_id % 32;
    beg += lane;

    for (unsigned int i = beg; i < end; i += 32) {
      unsigned int u = src[i];
      unsigned int v = dest[i];
      unsigned int w = weight[i];
      if (is_distance_infinity[u] == TRUE) {
        continue;
      }
      unsigned int temp_dist = distance[u] + w;
      if (distance[u] == -1) {
        continue;
      }
      if (temp_dist < distance[v]) {
        // relax
        //printf("%u %u\n", distance[v], temp_dist);
        int old_distance = atomicMin(&distance[v], temp_dist);
        atomicMin(&is_distance_infinity[v], FALSE);
        //printf("%u %u %u %d\n", old_distance, distance_cur[v], distance_prev[u] + w, is_distance_infinity[v]);
        // test for a change!
        if (old_distance != distance[v]) {
          //printf("there is change\n");
          atomicMin(noChange, FALSE);
        }
      }
    }
}

void puller(int blockSize, int blockNum,
  int sync, int smem, unsigned int *distance_cur,
  unsigned int *edges_src, unsigned int *edges_dest,
  unsigned int *edges_weight, unsigned int edges_length,
  unsigned int vertices_length){
    /* Will use these arrays instead of a vector
    * edges_src : array of all edges (indexed 0 to n) where the value is the vertex source index of the edge (since edges are directed)
    * edges_dest : same as above, except it tells the vertex destination index
    * edges_weight : same as above, except it tells the edge's weight
    * distance_prev : array of all vertices with their distance values
    * distance_cur : same as above
    */

    /* Allocate here... */
    unsigned int *distance_prev = (unsigned int *) malloc(vertices_length * sizeof(unsigned int));
    int *noChange = (int *) malloc(sizeof(int));
    int *is_distance_infinity = (int *) malloc(vertices_length * sizeof(int));

    *noChange = TRUE;

    unsigned int *cuda_edges_src, *cuda_edges_dest, *cuda_edges_weight;
    unsigned int *cuda_distance_prev, *cuda_distance_cur;
    int *cuda_noChange, *cuda_is_distance_infinity_prev, *cuda_is_distance_infinity_cur;

    // the distance to the first vertex is always 0
    distance_prev[0] = 0;
    distance_cur[0] = 0;
    is_distance_infinity[0] = FALSE;

    // setting an unsigned int to -1 will set it to the maximum value!
    for (int i = 1; i < vertices_length; i++) {
      distance_prev[i] = -1;
      distance_cur[i] = -1;
      is_distance_infinity[i] = TRUE;
    }

    cudaMalloc((void **)&cuda_edges_src, edges_length * sizeof(unsigned int));
    cudaMalloc((void **)&cuda_edges_dest, edges_length * sizeof(unsigned int));
    cudaMalloc((void **)&cuda_edges_weight, edges_length * sizeof(unsigned int));
    cudaMalloc((void **)&cuda_distance_prev, vertices_length * sizeof(unsigned int));
    cudaMalloc((void **)&cuda_distance_cur, vertices_length * sizeof(unsigned int));
    cudaMalloc((void **)&cuda_noChange, sizeof(int));
    cudaMalloc((void **)&cuda_is_distance_infinity_prev, vertices_length * sizeof(int));
    cudaMalloc((void **)&cuda_is_distance_infinity_cur, vertices_length * sizeof(int));

    cudaMemcpy(cuda_edges_src, edges_src, edges_length * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_edges_dest, edges_dest, edges_length * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_edges_weight, edges_weight, edges_length * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_distance_prev, distance_prev, vertices_length * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_distance_cur, distance_cur, vertices_length * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_noChange, noChange, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_is_distance_infinity_prev, is_distance_infinity, vertices_length * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_is_distance_infinity_cur, is_distance_infinity, vertices_length * sizeof(int), cudaMemcpyHostToDevice);

    setTime();

    /*
     * Do all the things here!
     **/
    // sync is out of core
    if (sync == 0) {
      // no shared memory
      if (smem == 0) {
        for (unsigned int i = 1; i < vertices_length; i++) {
          //printf("pass %u\n", i);
          edge_process_out_of_core_non_warp_partitioned<<<blockNum, blockSize>>>(edges_length, cuda_edges_src,
                                              cuda_edges_dest, cuda_edges_weight,
                                              cuda_distance_prev, cuda_distance_cur,
                                              cuda_noChange, cuda_is_distance_infinity_prev,
                                              cuda_is_distance_infinity_cur);
          cudaMemcpy(noChange, cuda_noChange, sizeof(int), cudaMemcpyDeviceToHost);
          if (*noChange == TRUE) break;
          *noChange = TRUE;
          cudaMemcpy(cuda_noChange, noChange, sizeof(int), cudaMemcpyHostToDevice);

          // get current distance and copy it to both cuda_distance_prev and cuda_distance_cur
          cudaMemcpy(distance_cur, cuda_distance_cur, vertices_length * sizeof(unsigned int), cudaMemcpyDeviceToHost);
          cudaMemcpy(cuda_distance_prev, distance_cur, vertices_length * sizeof(unsigned int), cudaMemcpyHostToDevice);
          cudaMemcpy(cuda_distance_cur, distance_cur, vertices_length * sizeof(unsigned int), cudaMemcpyHostToDevice);

          cudaMemcpy(is_distance_infinity, cuda_is_distance_infinity_cur, vertices_length * sizeof(unsigned int), cudaMemcpyDeviceToHost);
          cudaMemcpy(cuda_is_distance_infinity_prev, is_distance_infinity, vertices_length * sizeof(unsigned int), cudaMemcpyHostToDevice);
          cudaMemcpy(cuda_is_distance_infinity_cur, is_distance_infinity, vertices_length * sizeof(unsigned int), cudaMemcpyHostToDevice);

        }
      }
      // shared memory
      else if (smem == 1) {
        for (unsigned int i = 1; i < vertices_length; i++) {
          //printf("pass %d\n", i);
          edge_process_out_of_core_shared_memory<<<blockNum, blockSize, blockSize * sizeof(unsigned int)>>>(edges_length, cuda_edges_src,
                                              cuda_edges_dest, cuda_edges_weight,
                                              cuda_distance_prev, cuda_distance_cur,
                                              cuda_noChange, cuda_is_distance_infinity_prev,
                                              cuda_is_distance_infinity_cur);
          cudaMemcpy(noChange, cuda_noChange, sizeof(int), cudaMemcpyDeviceToHost);
          if (*noChange == TRUE) break;
          *noChange = TRUE;
          cudaMemcpy(cuda_noChange, noChange, sizeof(int), cudaMemcpyHostToDevice);

          // get current distance and copy it to both cuda_distance_prev and cuda_distance_cur
          cudaMemcpy(distance_cur, cuda_distance_cur, vertices_length * sizeof(unsigned int), cudaMemcpyDeviceToHost);
          cudaMemcpy(cuda_distance_prev, distance_cur, vertices_length * sizeof(unsigned int), cudaMemcpyHostToDevice);
          cudaMemcpy(cuda_distance_cur, distance_cur, vertices_length * sizeof(unsigned int), cudaMemcpyHostToDevice);

          cudaMemcpy(is_distance_infinity, cuda_is_distance_infinity_cur, vertices_length * sizeof(unsigned int), cudaMemcpyDeviceToHost);
          cudaMemcpy(cuda_is_distance_infinity_prev, is_distance_infinity, vertices_length * sizeof(unsigned int), cudaMemcpyHostToDevice);
          cudaMemcpy(cuda_is_distance_infinity_cur, is_distance_infinity, vertices_length * sizeof(unsigned int), cudaMemcpyHostToDevice);
        }
      }
      // no shared memory
      else {
        printf("No shared memory\n");
        exit(1);
      }

    }
    // sync is in core
    else if (sync == 1) {
      for (unsigned int i = 1; i < vertices_length; i++) {
        edge_process_in_core<<<blockNum, blockSize>>>(edges_length, vertices_length,
                                          cuda_edges_src, cuda_edges_dest,
                                          cuda_edges_weight, cuda_distance_cur,
                                          cuda_noChange, cuda_is_distance_infinity_prev);
        cudaMemcpy(noChange, cuda_noChange, sizeof(int), cudaMemcpyDeviceToHost);
        if (*noChange == TRUE) break;
        *noChange = TRUE;
        cudaMemcpy(cuda_noChange, noChange, sizeof(int), cudaMemcpyHostToDevice);
      }
    }

    else {
      // no syncing
      printf("No syncing tag\n");
      exit(1);
    }

    cudaDeviceSynchronize();
    std::cout << "Took " << getTime() << "ms.\n";

    cudaMemcpy(distance_cur, cuda_distance_cur, vertices_length * sizeof(unsigned int),
             cudaMemcpyDeviceToHost);

    /* Deallocate. */
    cudaFree(cuda_edges_src);
    cudaFree(cuda_edges_dest);
    cudaFree(cuda_edges_weight);
    cudaFree(cuda_distance_prev);
    cudaFree(cuda_distance_cur);
    cudaFree(cuda_noChange);
    cudaFree(cuda_is_distance_infinity_prev);
    cudaFree(cuda_is_distance_infinity_cur);

    free(edges_src);
    free(edges_dest);
    free(edges_weight);
    free(distance_prev);
    free(noChange);
    free(is_distance_infinity);
}
