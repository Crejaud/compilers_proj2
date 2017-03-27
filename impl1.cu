#include <vector>
#include <iostream>

#include "utils.h"
#include "cuda_error_check.cuh"
#include "initial_graph.hpp"
#include "parse_graph.hpp"

__global__ void edge_process(unsigned int edges_length,
                            unsigned int *src,
                            unsigned int *dest,
                            unsigned int *weight,
                            unsigned int *distance_prev,
                            unsigned int *distance_cur) {
    unsigned int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int thread_num = blockDim.x * gridDim.x;

    unsigned int warp_id = thread_id % 32 == 0 ? thread_id/32 : thread_id/32 + 1;
    unsigned int warp_num = thread_num % 32 == 0 ? thread_num/32 : thread_num/32 + 1;

    unsigned int load = edges_length % warp_num ? edges_length / warp_num + 1 : edges_length / warp_num;
    unsigned int beg = load * warp_id;
    unsigned int end = min(edges_length, beg + load);

    unsigned int i;
    for (i = beg; i < end; i += 32) {
      unsigned int u = src[i];
      unsigned int v = dest[i];
      unsigned int w = weight[i];
      if (distance_prev[u] + w < distance_prev[v])
        atomicMin(&distance_cur[v], distance_prev[u] + w);
    }
}

void puller(std::vector<initial_vertex> * peeps, int blockSize, int blockNum){
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
    unsigned int *distance_prev = (unsigned int *) malloc(peeps->size() * sizeof(unsigned int));
    unsigned int *distance_cur = (unsigned int *) malloc(peeps->size() * sizeof(unsigned int));

    unsigned int *cuda_edges_src, *cuda_edges_dest, *cuda_edges_weight;
    unsigned int *cuda_distance_prev, *cuda_distance_cur;

    // the distance to the first vertex is always 0
    distance_prev[0] = 0;
    distance_cur[0] = 0;

    // setting an unsigned int to -1 will set it to the maximum value!
    for (int i = 1; i < vertices_length; i++) {
      distance_prev[i] = -1;
      distance_cur[i] = -1;
    }

    // get edges_length
    for(std::vector<int>::size_type i = 0; i != peeps->size(); i++) {
      edges_length += peeps->at(i).nbrs.size();
    }

    // malloc edges arrays
    edges_src = (unsigned int *) malloc(edges_length * sizeof(unsigned int));
    edges_dest = (unsigned int *) malloc(edges_length * sizeof(unsigned int));
    edges_weight = (unsigned int *) malloc(edges_length * sizeof(unsigned int));


    int edge_index = 0;
    // get values for each array
    for(std::vector<int>::size_type i = 0; i != peeps->size(); i++) {
      for(std::vector<int>::size_type j = 0; j != peeps->at(i).nbrs.size(); j++) {
        edges_src[edge_index] = peeps->at(i).nbrs[j].srcIndex;
        edges_dest[edge_index] = i;
        edges_weight[edge_index] = peeps->at(i).nbrs[j].edgeValue.weight;

        edge_index++;
      }
    }

    cudaMalloc((void **)&cuda_edges_src, edges_length * sizeof(unsigned int));
    cudaMalloc((void **)&cuda_edges_dest, edges_length * sizeof(unsigned int));
    cudaMalloc((void **)&cuda_edges_weight, edges_length * sizeof(unsigned int));
    cudaMalloc((void **)&cuda_distance_prev, vertices_length * sizeof(unsigned int));
    cudaMalloc((void **)&cuda_distance_cur, vertices_length * sizeof(unsigned int));

    cudaMemcpy(cuda_edges_src, edges_src, edges_length * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_edges_dest, edges_dest, edges_length * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_edges_weight, edges_weight, edges_length * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_distance_prev, distance_prev, vertices_length * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_distance_cur, distance_cur, vertices_length * sizeof(unsigned int), cudaMemcpyHostToDevice);

    setTime();

    /*
     * Do all the things here!
     **/
    for (int i = 1; i < vertices_length; i++) {
    	edge_process<<<blockNum, blockSize>>>(edges_length, cuda_edges_src,
                                          cuda_edges_dest, cuda_edges_weight,
                                          cuda_distance_prev, cuda_distance_cur);
	if (noChange == 1) break;
	swap(cuda_distance_prev, cuda_distance_cur);
    }

    cudaDeviceSynchronize();
    std::cout << "Took " << getTime() << "ms.\n";

    cudaMemcpy(distance_cur, cuda_distance_cur, vertices_length * sizeof(unsigned int),
             cudaMemcpyDeviceToHost);

    // print it out to test
    for(int i = 0; i < vertices_length; i++) {
      printf("Vertex[%d] = %d\n", i, distance_cur[i]);
    }

    /* Deallocate. */
    cudaFree(cuda_edges_src);
    cudaFree(cuda_edges_dest);
    cudaFree(cuda_edges_weight);
    cudaFree(cuda_distance_prev);
    cudaFree(cuda_distance_cur);

    free(edges_src);
    free(edges_dest);
    free(edges_weight);
    free(distance_prev);
    free(distance_cur);
}
