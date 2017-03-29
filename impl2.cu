#include <vector>
#include <iostream>

#include "utils.h"
#include "cuda_error_check.cuh"
#include "initial_graph.hpp"
#include "parse_graph.hpp"

#define TRUE 1
#define FALSE 0

/* Work efficient edge process out of core with no shared memory */
__global__ void work_efficient_out_of_core(unsigned int edges_length,
                            unsigned int *src,
                            unsigned int *dest,
                            unsigned int *weight,
                            unsigned int *distance_prev,
                            unsigned int *distance_cur,
                            int *noChange,
                            int *is_distance_infinity_prev,
                            int *is_distance_infinity_cur,
                            unsigned int *mask,
                            unsigned int *num_edges_to_process,
                            unsigned int *warp_offsets,
                            unsigned int *T,
                            unsigned int *T_length) {
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
        //printf("%u %u %u %d\n", old_distance, distance_cur[v], distance_prev[u] + w, is_distance_infinity[v]);
        // test for a change!
        if (old_distance != distance_cur[v]) {
          //printf("there is change\n");
          atomicMin(noChange, FALSE);
        }
      }
      atomicOr(&mask[warp_id], __ballot(distance_cur[u] != distance_prev[u]));
    }
    __syncthreads();

    // set the number of edges to process for a warps
    atomicMax(&num_edges_to_process[warp_id], __popc(mask[warp_id]));
}

/* This kernel function will perform block level parallel prefix sum to get
 * the offset of every warp's first to-process edge in the final to-process edge
 * list T */
__global__ void filtering(int edges_length,
                          unsigned int *num_edges_to_process,
                          unsigned int *warp_offsets,
                          unsigned int *distance_prev,
                          unsigned int *distance_cur,
                          unsigned int *T,
                          unsigned int *T_length,
                          unsigned int *src) {
  extern __shared__ unsigned int smem_warp_offsets[ ];
  // we can assume it will fit since there will be at most 64 warps
  unsigned int warp_id = threadIdx.x;

  __syncthreads();

  int offset = 1;

  if (threadIdx.x*2+1 < blockDim.x) {
    // do parallel prefix sum!

    smem_warp_offsets[2*threadIdx.x] = num_edges_to_process[2*threadIdx.x];
    smem_warp_offsets[2*threadIdx.x+1] = num_edges_to_process[2*threadIdx.x+1];

    for (int d = blockDim.x>>1; d > 0; d >>= 1) {
      __syncthreads();

      if (threadIdx.x < d) {
        int ai = offset*(2*threadIdx.x+1)-1;
        int bi = offset*(2*threadIdx.x+2)-1;

        smem_warp_offsets[bi] += smem_warp_offsets[ai];
      }
      offset *= 2;
    }

    if (threadIdx.x == 0) {
      smem_warp_offsets[blockDim.x - 1] = 0;
    }

    for (int d = 1; d < blockDim.x; d *= 2) {
      offset >>= 1;
      __syncthreads();

      if (threadIdx.x < d) {
        int ai = offset*(2*threadIdx.x+1)-1;
        int bi = offset*(2*threadIdx.x+2)-1;

        unsigned int t = smem_warp_offsets[ai];
        smem_warp_offsets[ai] = smem_warp_offsets[bi];
        smem_warp_offsets[bi] += t;
      }
    }

    __syncthreads();

    warp_offsets[2*threadIdx.x] = smem_warp_offsets[2*threadIdx.x];
    warp_offsets[2*threadIdx.x+1] = smem_warp_offsets[2*threadIdx.x + 1];
  }

  __syncthreads();

  // the new length of T is the total number of edges to process!
  printf("blockDim = %u | T becomes %u, since %u and %u | warp_id = %u\n", blockDim.x, *T_length, warp_offsets[blockDim.x-1], num_edges_to_process[blockDim.x-1], threadIdx.x);
  *T_length = warp_offsets[blockDim.x-1] + num_edges_to_process[blockDim.x-1];

  // parallel prefix sum is done and warp_offsets is complete

  // now we must create T

  unsigned int cur_offset = warp_offsets[threadIdx.x];

  unsigned int load = edges_length % blockDim.x == 0 ? edges_length / blockDim.x : edges_length / blockDim.x + 1;
  unsigned int beg = load * threadIdx.x;
  unsigned int end = min(edges_length, beg + load);

  for (unsigned int i = beg; i < end; i++) {
    // done with filling in T
    if (cur_offset >= warp_offsets[threadIdx.x] + num_edges_to_process[threadIdx.x])
      return;

    // if they're the same
    if (distance_cur[src[i]] != distance_prev[src[i]]) {
      T[cur_offset] = i;
      cur_offset++;
    }
  }
}

/* Work efficient edge process in core */
__global__ void work_efficient_in_core(unsigned int edges_length,
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

void neighborHandler(std::vector<initial_vertex> * peeps, int blockSize, int blockNum, int sync, int smem, unsigned int *distance_cur){
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
  unsigned int *mask, *num_edges_to_process;
  int *noChange = (int *) malloc(sizeof(int));
  int *is_distance_infinity = (int *) malloc(vertices_length * sizeof(int));
  unsigned int *T, *T_length;
  unsigned int *warp_offsets;

  // this is the total number of warps
  unsigned int warp_num = blockSize * blockNum % 32 == 0 ? blockSize * blockNum / 32 : blockSize * blockNum / 32 + 1;

  mask = (unsigned int *) malloc( warp_num * sizeof(unsigned int));
  num_edges_to_process = (unsigned int *) malloc( warp_num * sizeof(unsigned int));
  warp_offsets = (unsigned int *) malloc( warp_num * sizeof(unsigned int));

  for(int i = 0; i < warp_num; i++) {
    mask[i] = 0;
    num_edges_to_process[i] = 0;
    warp_offsets[i] = 0;
  }

  *noChange = TRUE;

  unsigned int *cuda_edges_src, *cuda_edges_dest, *cuda_edges_weight;
  unsigned int *cuda_distance_prev, *cuda_distance_cur;
  unsigned int *cuda_T, *cuda_T_length;
  unsigned int *cuda_mask, *cuda_num_edges_to_process;
  unsigned int *cuda_warp_offsets;
  int *cuda_noChange, *cuda_is_distance_infinity_prev,
    *cuda_is_distance_infinity_cur;

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
  T = (unsigned int *) malloc(edges_length * sizeof(unsigned int));
  T_length = (unsigned int *) malloc(sizeof(unsigned int));


  int edge_index = 0;
  // get values for each array
  for(std::vector<int>::size_type i = 0; i != vertices_length; i++) {
    for(std::vector<int>::size_type j = 0; j != peeps->at(i).nbrs.size(); j++) {
      edges_src[edge_index] = peeps->at(i).nbrs[j].srcIndex;
      edges_dest[edge_index] = i;
      edges_weight[edge_index] = peeps->at(i).nbrs[j].edgeValue.weight;
      // initially set should_update_edges to true if the source is at 0, since everything
      // else will have infinite distance.
      if (edges_src[edge_index] == 0) {
        unsigned int load = edges_length % warp_num == 0 ? edges_length / warp_num : edges_length / warp_num + 1;
        unsigned int warp_id = edge_index / load;
        num_edges_to_process[warp_id];
      }
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
  cudaMalloc((void **)&cuda_is_distance_infinity_prev, vertices_length * sizeof(int));
  cudaMalloc((void **)&cuda_is_distance_infinity_cur, vertices_length * sizeof(int));
  cudaMalloc((void **)&cuda_T, edges_length * sizeof(unsigned int));
  cudaMalloc((void **)&cuda_T_length, sizeof(unsigned int));
  cudaMalloc((void **)&cuda_mask, warp_num * sizeof(unsigned int));
  cudaMalloc((void **)&cuda_num_edges_to_process, warp_num * sizeof(unsigned int));
  cudaMalloc((void **)&cuda_warp_offsets, warp_num * sizeof(unsigned int));

  cudaMemcpy(cuda_edges_src, edges_src, edges_length * sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_edges_dest, edges_dest, edges_length * sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_edges_weight, edges_weight, edges_length * sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_distance_prev, distance_prev, vertices_length * sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_distance_cur, distance_cur, vertices_length * sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_noChange, noChange, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_is_distance_infinity_prev, is_distance_infinity, vertices_length * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_is_distance_infinity_cur, is_distance_infinity, vertices_length * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_T, T, edges_length * sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_T_length, T_length, sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_mask, mask, warp_num * sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_num_edges_to_process, num_edges_to_process, warp_num * sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_warp_offsets, warp_offsets, warp_num * sizeof(unsigned int), cudaMemcpyHostToDevice);


  setTime();

  /*
   * Do all the things here!
   **/
  // sync is out of core
  if (sync == 0) {
    for (unsigned int i = 1; i < vertices_length; i++) {
      printf("pass %u, starting filtering\n", i);
      filtering<<<1, warp_num, warp_num * sizeof(unsigned int)>>>(edges_length,
                                cuda_num_edges_to_process,
                                cuda_warp_offsets,
                                cuda_distance_prev,
                                cuda_distance_cur,
                                cuda_T,
                                cuda_T_length,
                                cuda_edges_src);

      printf("filtering done\n");

      cudaMemcpy(T, cuda_T, edges_length * sizeof(unsigned int), cudaMemcpyDeviceToHost);
      cudaMemcpy(T_length, cuda_T_length, sizeof(unsigned int), cudaMemcpyDeviceToHost);

      printf("T_length = %u\n", *T_length);
      //print out T for testing
      for (unsigned int j = 0; j < *T_length; j++) {
        printf("T[%u] = %u\n", j, T[j]);
      }

      // reset these values back to 0
      for(unsigned int j = 0; j < warp_num; j++) {
        mask[j] = 0;
        num_edges_to_process[j] = 0;
      }

      printf("past forloop\n");

      printf("past nochange reset\n");

      // get current distance and copy it to both cuda_distance_prev and cuda_distance_cur
      cudaMemcpy(distance_cur, cuda_distance_cur, vertices_length * sizeof(unsigned int), cudaMemcpyDeviceToHost);
      cudaMemcpy(cuda_distance_prev, distance_cur, vertices_length * sizeof(unsigned int), cudaMemcpyHostToDevice);
      cudaMemcpy(cuda_distance_cur, distance_cur, vertices_length * sizeof(unsigned int), cudaMemcpyHostToDevice);

      printf("past distance reset\n");

      cudaMemcpy(is_distance_infinity, cuda_is_distance_infinity_cur, vertices_length * sizeof(unsigned int), cudaMemcpyDeviceToHost);
      cudaMemcpy(cuda_is_distance_infinity_prev, is_distance_infinity, vertices_length * sizeof(unsigned int), cudaMemcpyHostToDevice);
      cudaMemcpy(cuda_is_distance_infinity_cur, is_distance_infinity, vertices_length * sizeof(unsigned int), cudaMemcpyHostToDevice);

      printf("starting outcore\n");

      //printf("pass %u\n", i);
      work_efficient_out_of_core<<<blockNum, blockSize>>>(edges_length, cuda_edges_src,
                                          cuda_edges_dest, cuda_edges_weight,
                                          cuda_distance_prev, cuda_distance_cur,
                                          cuda_noChange, cuda_is_distance_infinity_prev,
                                          cuda_is_distance_infinity_cur, cuda_mask,
                                          cuda_num_edges_to_process,
                                          cuda_warp_offsets,
                                          cuda_T, cuda_T_length);

      printf("outcore done\n");

      cudaMemcpy(noChange, cuda_noChange, sizeof(int), cudaMemcpyDeviceToHost);
      if (*noChange == TRUE) break;
      *noChange = TRUE;
      cudaMemcpy(cuda_noChange, noChange, sizeof(int), cudaMemcpyHostToDevice);

      printf("there was change\n");
    }
  }
  // sync is in core
  else if (sync == 1) {
    for (unsigned int i = 1; i < vertices_length; i++) {
      work_efficient_in_core<<<blockNum, blockSize>>>(edges_length, vertices_length,
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
  cudaFree(cuda_mask);
  cudaFree(cuda_num_edges_to_process);
  cudaFree(cuda_T);
  cudaFree(cuda_T_length);

  free(edges_src);
  free(edges_dest);
  free(edges_weight);
  free(distance_prev);
  free(noChange);
  free(is_distance_infinity);
  free(mask);
  free(num_edges_to_process);
  free(T);
  free(T_length);
}
