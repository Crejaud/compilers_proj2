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
                            int *is_distance_infinity) {
    extern __shared__ unsigned int s_data[ ];

    unsigned int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int thread_num = blockDim.x * gridDim.x;

    unsigned int warp_id = thread_id % 32 == 0 ? thread_id/32 : thread_id/32 + 1;
    unsigned int warp_num = thread_num % 32 == 0 ? thread_num/32 : thread_num/32 + 1;

    unsigned int load = edges_length % warp_num ? edges_length / warp_num + 1 : edges_length / warp_num;
    unsigned int beg = load * warp_id;
    unsigned int end = min(edges_length, beg + load);
    unsigned int lane = thread_id % 32;
    beg += lane;

    unsigned int i;
    for (i = beg; i < end; i += 32) {
      unsigned int u = src[i];
      unsigned int v = dest[i];
      unsigned int w = weight[i];

      if (is_distance_infinity[u] == TRUE) {
        s_data[threadIdx.x] = -1;
      }
      else {
        s_data[threadIdx.x] = min(distance_cur[v], distance_prev[u] + w);
      }

      printf("s_data at %u is %u, lane %u, i %u\n", warp_id, s_data[threadIdx.x], lane, i);

      __syncthreads();

      // segmented scan to find minimum
      if (lane >= 1 && dest[i] == dest[i - 1])
        s_data[threadIdx.x] = min(s_data[threadIdx.x], s_data[threadIdx.x-1]);
      if (lane >= 2 && dest[i] == dest[i - 2])
        s_data[threadIdx.x] = min(s_data[threadIdx.x], s_data[threadIdx.x-2]);
      if (lane >= 4 && dest[i] == dest[i - 4])
        s_data[threadIdx.x] = min(s_data[threadIdx.x], s_data[threadIdx.x-4]);
      if (lane >= 8 && dest[i] == dest[i - 8])
        s_data[threadIdx.x] = min(s_data[threadIdx.x], s_data[threadIdx.x-8]);
      if (lane >= 16 && dest[i] == dest[i - 16])
        s_data[threadIdx.x] = min(s_data[threadIdx.x], s_data[threadIdx.x-16]);

      __syncthreads();

      // i is in bounds
      if (i + 1 < edges_length) {
        // this thread is the last thread for the segment, so it holds the min
        if (dest[i] != dest[i + 1]) {
          printf("the min for dest %u is %u\n", dest[i], s_data[threadIdx.x]);
          int old_distance = atomicMin(&distance_cur[v], s_data[threadIdx.x]);
          atomicMin(&is_distance_infinity[v], FALSE);
          // test for a change!
          if (old_distance != distance_cur[v]) {
            //printf("there is change\n");
            *noChange = FALSE;
          }
        }
      }
      // i is the last element
      else {
        printf("the min for dest %u is %u\n", dest[i], s_data[threadIdx.x]);
        int old_distance = atomicMin(&distance_cur[v], s_data[threadIdx.x]);
        atomicMin(&is_distance_infinity[v], FALSE);
        // test for a change!
        if (old_distance != distance_cur[v]) {
          //printf("there is change\n");
          *noChange = FALSE;
        }
      }
    }
}

/* Edge process out of core with no shared memory */
__global__ void edge_process_out_of_core(unsigned int edges_length,
                            unsigned int *src,
                            unsigned int *dest,
                            unsigned int *weight,
                            unsigned int *distance_prev,
                            unsigned int *distance_cur,
                            int *noChange,
                            int *is_distance_infinity) {
    unsigned int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int thread_num = blockDim.x * gridDim.x;

    unsigned int warp_id = thread_id % 32 == 0 ? thread_id/32 : thread_id/32 + 1;
    unsigned int warp_num = thread_num % 32 == 0 ? thread_num/32 : thread_num/32 + 1;

    unsigned int load = edges_length % warp_num ? edges_length / warp_num + 1 : edges_length / warp_num;
    unsigned int beg = load * warp_id;
    unsigned int end = min(edges_length, beg + load);
    unsigned int lane = thread_id % 32;
    beg += lane;

    unsigned int i;
    for (i = beg; i < end; i += 32) {
      unsigned int u = src[i];
      unsigned int v = dest[i];
      unsigned int w = weight[i];
      if (is_distance_infinity[u] == TRUE) {
        break;
      }
      if (distance_prev[u] + w < distance_prev[v]) {
        // relax
        //printf("%u %u\n", distance_cur[v], distance_prev[u] + w);
        int old_distance = atomicMin(&distance_cur[v], distance_prev[u] + w);
        atomicMin(&is_distance_infinity[v], FALSE);
        //printf("%u %u %u %d\n", old_distance, distance_cur[v], distance_prev[u] + w, is_distance_infinity[v]);
        // test for a change!
        if (old_distance != distance_cur[v]) {
          //printf("there is change\n");
          *noChange = FALSE;
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

    unsigned int warp_id = thread_id % 32 == 0 ? thread_id/32 : thread_id/32 + 1;
    unsigned int warp_num = thread_num % 32 == 0 ? thread_num/32 : thread_num/32 + 1;

    unsigned int load = edges_length % warp_num ? edges_length / warp_num + 1 : edges_length / warp_num;
    unsigned int beg = load * warp_id;
    unsigned int end = min(edges_length, beg + load);

    unsigned int i;
    for (unsigned int j = 1; j < vertices_length; j++) {
      __syncthreads();
      for (i = beg; i < end; i += 32) {
        unsigned int u = src[i];
        unsigned int v = dest[i];
        unsigned int w = weight[i];
        if (is_distance_infinity[u] == TRUE) {
          break;
        }
        unsigned int temp_dist = distance[u] + w;
        if (temp_dist < distance[v]) {
          // relax
          //printf("%u %u\n", distance_cur[v], distance_prev[u] + w);
          int old_distance = atomicMin(&distance[v], distance[u] + w);
          atomicMin(&is_distance_infinity[v], FALSE);
          //printf("%u %u %u %d\n", old_distance, distance_cur[v], distance_prev[u] + w, is_distance_infinity[v]);
          // test for a change!
          if (old_distance != distance[v]) {
            //printf("there is change\n");
            *noChange = FALSE;
          }
        }
      }
    }
}

void puller(std::vector<initial_vertex> * peeps, int blockSize, int blockNum, int sync, int smem, unsigned int *distance_cur){
    /* Will use these arrays instead of a vector
    * edges_src : array of all edges (indexed 0 to n) where the value is the vertex source index of the edge (since edges are directed)
    * edges_dest : same as above, except it tells the vertex destination index
    * edges_weight : same as above, except it tells the edge's weight
    * distance_prev : array of all vertices with their distance values
    * distance_cur : same as above
    */

    /* Allocate here... */
    unsigned int *edges_src, *edges_dest, *edges_weight;
    unsigned int edges_length = 0;
    unsigned int vertices_length = peeps->size();
    unsigned int *distance_prev = (unsigned int *) malloc(vertices_length * sizeof(unsigned int));
    int *noChange = (int *) malloc(sizeof(int));
    int *is_distance_infinity = (int *) malloc(vertices_length * sizeof(int));

    *noChange = TRUE;

    unsigned int *cuda_edges_src, *cuda_edges_dest, *cuda_edges_weight;
    unsigned int *cuda_distance_prev, *cuda_distance_cur;
    int *cuda_noChange, *cuda_is_distance_infinity;

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

    // get edges_length
    for(std::vector<int>::size_type i = 0; i != vertices_length; i++) {
      edges_length += peeps->at(i).nbrs.size();
    }

    // malloc edges arrays
    edges_src = (unsigned int *) malloc(edges_length * sizeof(unsigned int));
    edges_dest = (unsigned int *) malloc(edges_length * sizeof(unsigned int));
    edges_weight = (unsigned int *) malloc(edges_length * sizeof(unsigned int));


    int edge_index = 0;
    // get values for each array
    for(std::vector<int>::size_type i = 0; i != vertices_length; i++) {
      for(std::vector<int>::size_type j = 0; j != peeps->at(i).nbrs.size(); j++) {
        edges_src[edge_index] = peeps->at(i).nbrs[j].srcIndex;
        edges_dest[edge_index] = i;
        edges_weight[edge_index] = peeps->at(i).nbrs[j].edgeValue.weight;
        //printf("src: %u | dest: %u | weight: %u\n", edges_src[edge_index], edges_dest[edge_index], edges_weight[edge_index]);

        edge_index++;
      }
    }

    cudaMalloc((void **)&cuda_edges_src, edges_length * sizeof(unsigned int));
    cudaMalloc((void **)&cuda_edges_dest, edges_length * sizeof(unsigned int));
    cudaMalloc((void **)&cuda_edges_weight, edges_length * sizeof(unsigned int));
    cudaMalloc((void **)&cuda_distance_prev, vertices_length * sizeof(unsigned int));
    cudaMalloc((void **)&cuda_distance_cur, vertices_length * sizeof(unsigned int));
    cudaMalloc((void **)&cuda_noChange, sizeof(int));
    cudaMalloc((void **)&cuda_is_distance_infinity, vertices_length * sizeof(int));

    cudaMemcpy(cuda_edges_src, edges_src, edges_length * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_edges_dest, edges_dest, edges_length * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_edges_weight, edges_weight, edges_length * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_distance_prev, distance_prev, vertices_length * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_distance_cur, distance_cur, vertices_length * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_noChange, noChange, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_is_distance_infinity, is_distance_infinity, vertices_length * sizeof(int), cudaMemcpyHostToDevice);

    setTime();

    /*
     * Do all the things here!
     **/
    // sync is out of core
    if (sync == 0) {
      // no shared memory
      if (smem == 0) {
        for (int i = 1; i < vertices_length; i++) {
          edge_process_out_of_core<<<blockNum, blockSize>>>(edges_length, cuda_edges_src,
                                              cuda_edges_dest, cuda_edges_weight,
                                              cuda_distance_prev, cuda_distance_cur,
                                              cuda_noChange, cuda_is_distance_infinity);
          cudaMemcpy(noChange, cuda_noChange, sizeof(int), cudaMemcpyDeviceToHost);
          if (*noChange == TRUE) break;
          *noChange = TRUE;
          cudaMemcpy(cuda_noChange, noChange, sizeof(int), cudaMemcpyHostToDevice);

          // get current distance and copy it to both cuda_distance_prev and cuda_distance_cur
          cudaMemcpy(distance_cur, cuda_distance_cur, vertices_length * sizeof(unsigned int), cudaMemcpyDeviceToHost);
          cudaMemcpy(cuda_distance_prev, distance_cur, vertices_length * sizeof(unsigned int), cudaMemcpyHostToDevice);
          cudaMemcpy(cuda_distance_cur, distance_cur, vertices_length * sizeof(unsigned int), cudaMemcpyHostToDevice);
        }
      }
      // shared memory
      else if (smem == 1) {
        for (int i = 1; i < vertices_length; i++) {
          printf("pass %d\n", i);
          edge_process_out_of_core_shared_memory<<<blockNum, blockSize, blockSize * sizeof(unsigned int)>>>(edges_length, cuda_edges_src,
                                              cuda_edges_dest, cuda_edges_weight,
                                              cuda_distance_prev, cuda_distance_cur,
                                              cuda_noChange, cuda_is_distance_infinity);
          cudaMemcpy(noChange, cuda_noChange, sizeof(int), cudaMemcpyDeviceToHost);
          if (*noChange == TRUE) break;
          *noChange = TRUE;
          cudaMemcpy(cuda_noChange, noChange, sizeof(int), cudaMemcpyHostToDevice);

          // get current distance and copy it to both cuda_distance_prev and cuda_distance_cur
          cudaMemcpy(distance_cur, cuda_distance_cur, vertices_length * sizeof(unsigned int), cudaMemcpyDeviceToHost);
          cudaMemcpy(cuda_distance_prev, distance_cur, vertices_length * sizeof(unsigned int), cudaMemcpyHostToDevice);
          cudaMemcpy(cuda_distance_cur, distance_cur, vertices_length * sizeof(unsigned int), cudaMemcpyHostToDevice);
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
      edge_process_in_core<<<blockNum, blockSize>>>(edges_length, vertices_length,
                                          cuda_edges_src, cuda_edges_dest,
                                          cuda_edges_weight, cuda_distance_cur,
                                          cuda_noChange, cuda_is_distance_infinity);
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
    cudaFree(cuda_is_distance_infinity);

    free(edges_src);
    free(edges_dest);
    free(edges_weight);
    free(distance_prev);
    free(noChange);
    free(is_distance_infinity);
}
