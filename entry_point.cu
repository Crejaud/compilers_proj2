#include <cstring>
#include <stdexcept>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

#include "utils.h"
#include "cuda_error_check.cuh"
#include "initial_graph.hpp"
#include "parse_graph.hpp"

#include "opt.cu"
#include "impl2.cu"
#include "impl1.cu"

enum class ProcessingType {Push, Neighbor, Own, Unknown};
enum SyncMode {InCore, OutOfCore};
enum SmemMode {UseSmem, UseNoSmem};
enum SyncMode syncMethod;
enum SmemMode smemMethod;

int sync = -1;
int smem = -1;

void mergeSort(unsigned int *arr, unsigned int *other, unsigned int *weight, unsigned int l, unsigned int r);
void merge(unsigned int *arr, unsigned int *other, unsigned int *weight, unsigned int l, unsigned int m, unsigned int r);

// Open files safely.
template <typename T_file>
void openFileToAccess( T_file& input_file, std::string file_name ) {
	input_file.open( file_name.c_str() );
	if( !input_file )
		throw std::runtime_error( "Failed to open specified file: " + file_name + "\n" );
}


// Execution entry point.
int main( int argc, char** argv )
{

	std::string usage =
		"\tRequired command line arguments:\n\
			Input file: E.g., --input in.txt\n\
                        Block size: E.g., --bsize 512\n\
                        Block count: E.g., --bcount 192\n\
                        Output path: E.g., --output output.txt\n\
			Processing method: E.g., --method bmf (bellman-ford), or tpe (to-process-edge), or opt (one further optimizations)\n\
			Shared memory usage: E.g., --usesmem yes, or no \n\
			Sync method: E.g., --sync incore, or outcore\n\
			Sort method: E.g., --sort_by_dest yes, or no\n";

	try {

		std::ifstream inputFile;
		std::ofstream outputFile;
		int selectedDevice = 0;
		int bsize = 0, bcount = 0;
		int vwsize = 32;
		int threads = 1;
		long long arbparam = 0;
		bool nonDirectedGraph = false;		// By default, the graph is directed.
		ProcessingType processingMethod = ProcessingType::Unknown;
		syncMethod = OutOfCore;


		/********************************
		 * GETTING INPUT PARAMETERS.
		 ********************************/

		for( int iii = 1; iii < argc; ++iii )
			if ( !strcmp(argv[iii], "--method") && iii != argc-1 ) {
				if ( !strcmp(argv[iii+1], "bmf") )
				        processingMethod = ProcessingType::Push;
				else if ( !strcmp(argv[iii+1], "tpe") )
    				        processingMethod = ProcessingType::Neighbor;
				else if ( !strcmp(argv[iii+1], "opt") )
				    processingMethod = ProcessingType::Own;
				else{
           std::cerr << "\n Un-recognized method parameter value \n\n";
           exit;
         }
			}
			else if ( !strcmp(argv[iii], "--sync") && iii != argc-1 ) {
				if ( !strcmp(argv[iii+1], "incore") ) {
				        syncMethod = InCore;
								sync = 1;
							}
				else if ( !strcmp(argv[iii+1], "outcore") ) {
    				        syncMethod = OutOfCore;
										sync = 0;
									}
				else{
           std::cerr << "\n Un-recognized sync parameter value \n\n";
           exit;
         }

			}
			else if ( !strcmp(argv[iii], "--usesmem") && iii != argc-1 ) {
				if ( !strcmp(argv[iii+1], "yes") ) {
				        smemMethod = UseSmem;
								smem = 1;
							}
				else if ( !strcmp(argv[iii+1], "no") ) {
    				        smemMethod = UseNoSmem;
										smem = 0;
									}
        else{
           std::cerr << "\n Un-recognized usesmem parameter value \n\n";
           exit;
         }
			}
			int shouldSortByDestination = 0;
			else if ( !strcmp(argv[iii], "--sort_by_dest") && iii != argc-1 ) {
				if ( !strcmp(argv[iii+1], "yes") ) {
				        shouldSortByDestination = 1;
							}
				else if ( !strcmp(argv[iii+1], "no") ) {
    				        shouldSortByDestination = 0;
									}
        else{
           std::cerr << "\n Un-recognized sort_by_dest parameter value \n\n";
           exit;
         }
			}
			else if( !strcmp( argv[iii], "--input" ) && iii != argc-1 /*is not the last one*/)
				openFileToAccess< std::ifstream >( inputFile, std::string( argv[iii+1] ) );
			else if( !strcmp( argv[iii], "--output" ) && iii != argc-1 /*is not the last one*/)
				openFileToAccess< std::ofstream >( outputFile, std::string( argv[iii+1] ) );
			else if( !strcmp( argv[iii], "--bsize" ) && iii != argc-1 /*is not the last one*/)
				bsize = std::atoi( argv[iii+1] );
			else if( !strcmp( argv[iii], "--bcount" ) && iii != argc-1 /*is not the last one*/)
				bcount = std::atoi( argv[iii+1] );

		if(bsize <= 0 || bcount <= 0){
			std::cerr << "Usage: " << usage;
      exit;
			throw std::runtime_error("\nAn initialization error happened.\nExiting.");
		}
		if( !inputFile.is_open() || processingMethod == ProcessingType::Unknown ) {
			std::cerr << "Usage: " << usage;
			throw std::runtime_error( "\nAn initialization error happened.\nExiting." );
		}
		if( !outputFile.is_open() )
			openFileToAccess< std::ofstream >( outputFile, "out.txt" );
		CUDAErrorCheck( cudaSetDevice( selectedDevice ) );
		std::cout << "Device with ID " << selectedDevice << " is selected to process the graph.\n";


		/********************************
		 * Read the input graph file.
		 ********************************/

		std::cout << "Collecting the input graph ...\n";
		std::vector<initial_vertex> parsedGraph( 0 );
		uint nEdges = parse_graph::parse(
				inputFile,		// Input file.
				parsedGraph,	// The parsed graph.
				arbparam,
				nonDirectedGraph );		// Arbitrary user-provided parameter.
		std::cout << "Input graph collected with " << parsedGraph.size() << " vertices and " << nEdges << " edges.\n";


		/********************************
		 * Process the graph.
		 ********************************/

		unsigned int * distance = (unsigned int *) malloc(parsedGraph.size() * sizeof(unsigned int));

		unsigned int *edges_src, *edges_dest, *edges_weight;
    unsigned int edges_length = 0;
		unsigned int vertices_length = parsedGraph.size();

    // get edges_length
    for(std::vector<int>::size_type i = 0; i != vertices_length; i++) {
      edges_length += parsedGraph.at(i).nbrs.size();
    }

    // malloc edges arrays
    edges_src = (unsigned int *) malloc(edges_length * sizeof(unsigned int));
    edges_dest = (unsigned int *) malloc(edges_length * sizeof(unsigned int));
    edges_weight = (unsigned int *) malloc(edges_length * sizeof(unsigned int));


    int edge_index = 0;
    // get values for each array
    for(std::vector<int>::size_type i = 0; i != vertices_length; i++) {
      for(std::vector<int>::size_type j = 0; j != parsedGraph.at(i).nbrs.size(); j++) {
        edges_src[edge_index] = parsedGraph.at(i).nbrs[j].srcIndex;
        edges_dest[edge_index] = i;
        edges_weight[edge_index] = parsedGraph.at(i).nbrs[j].edgeValue.weight;
        //printf("src: %u | dest: %u | weight: %u\n", edges_src[edge_index], edges_dest[edge_index], edges_weight[edge_index]);

        edge_index++;
      }
    }

		// sort the edges by destination
		if (shouldSortByDestination == 1) {
			mergeSort(edges_dest, edges_src, edges_weight, 0, edges_length - 1);
		}
		// sort the edges by source
		else {
			mergeSort(edges_src, edges_dest, edges_weight, 0, edges_length - 1);
		}

		switch(processingMethod){
		case ProcessingType::Push:
		    puller(bsize, bcount, sync, smem, distance,
					edges_src, edges_dest, edges_weight, edges_length, vertices_length);
		    break;
		case ProcessingType::Neighbor:
		    neighborHandler(bsize, bcount, sync, smem, distance,
					edges_src, edges_dest, edges_weight, edges_length, vertices_length);
		    break;
		default:
		    own(&parsedGraph, bsize, bcount);
		}

		// print it out to test
    char outputStr[100];
    for(int i = 0; i < parsedGraph.size(); i++) {
      sprintf(outputStr, "%u:%u\n", i, distance[i]);
      outputFile << outputStr;
    }

		/********************************
		 * It's done here.
		 ********************************/

		CUDAErrorCheck( cudaDeviceReset() );
		std::cout << "Done.\n";
		return( EXIT_SUCCESS );

	}
	catch( const std::exception& strException ) {
		std::cerr << strException.what() << "\n";
		return( EXIT_FAILURE );
	}
	catch(...) {
		std::cerr << "An exception has occurred." << std::endl;
		return( EXIT_FAILURE );
	}

}

void mergeSort(unsigned int *arr, unsigned int *other, unsigned int *weight, unsigned int l, unsigned int r) {
	if (l < r) {
		unsigned int m = l + (r - l)/2;

		mergeSort(arr, other, weight, l, m);
		mergeSort(arr, other, weight, m+1, r);

		merge(arr, other, weight, l, m, r);
	}
}

void merge(unsigned int *arr, unsigned int *other, unsigned int *weight, unsigned int l, unsigned int m, unsigned int r) {
	unsigned int i, j, k;
	unsigned int n1 = m - l + 1;
	unsigned int n2 = r - m;

	unsigned int *L1 = (unsigned int *) malloc(n1 * sizeof(unsigned int));
	unsigned int *L2 = (unsigned int *) malloc(n1 * sizeof(unsigned int));
	unsigned int *L3 = (unsigned int *) malloc(n1 * sizeof(unsigned int));

	unsigned int *R1 = (unsigned int *) malloc(n2 * sizeof(unsigned int));
	unsigned int *R2 = (unsigned int *) malloc(n2 * sizeof(unsigned int));
	unsigned int *R3 = (unsigned int *) malloc(n2 * sizeof(unsigned int));

	for (i = 0; i < n1; i++) {
		L1[i] = arr[l + i];
		L2[i] = other[l + i];
		L3[i] = weight[l + i];
	}
	for (j = 0; j < n2; j++) {
		R1[j] = arr[m + j + 1];
		R2[j] = other[m + j + 1];
		R3[j] = weight[m + j + 1];
	}

	i = 0;
	j = 0;
	k = l;
	while (i < n1 && j < n2) {
		if (L1[i] <= R1[j]) {
			arr[k] = L1[i];
			other[k] = L2[i];
			weight[k] = L3[i];
			i++;
		}
		else {
			arr[k] = R1[j];
			other[k] = R2[j];
			weight[k] = R3[j];
			j++;
		}
		k++;
	}

	while (i < n1) {
		arr[k] = L1[i];
		other[k] = L2[i];
		weight[k] = L3[i];
		i++;
		k++;
	}

	while (j < n2) {
		arr[k] = R1[j];
		other[k] = R2[j];
		weight[k] = R3[j];
		j++;
		k++;
	}

	free(L1);
	free(L2);
	free(L3);
	free(R1);
	free(R2);
	free(R3);
}
