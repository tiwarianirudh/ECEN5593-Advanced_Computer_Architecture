# ECEN5593-Advanced_Computer_Architecture #

## Project: Implementation and Analysis of Dijkstra’s Algorithm and Floyd Warshall’s Algorithm: OpenMP and CUDA ##

### Authors: Anirudh Tiwari & Vishal Vishnani###

####Date: May 10th, 2017####

####Supported Hardware: NVIDIA JETSON TX1####

####File Structure:####
1.	Dijkstra.c: Consists of Serial and OpenMP implementation of Dijkstra’s Algorithm
2.	Dijkstra.cu: Consists of Serial and CUDA implementation of Dijkstra’s Algorithm
3.	Floyd_Warshall.c: Consists of Serial and OpenMP implementation of Dijkstra’s Algorithm
4.	Floyd_Warshall.cu: Consists of Serial and CUDA implementation of Dijkstra’s Algorithm
5.	Makefile: To run the above files

####Project Execution:####
*Dijkstra’s Algorithm: 
  *To run Serial vs OpenMP
    *	make Dijkstra_CPU.out
    * ./Dijkstra_CPU.out
  *To run CPU vs GPU
    *	make Dijkstra_GPU.out
    *	./Dijkstra_GPU.out

*Floyd Warshall’s Algorithm
  *To run Serial vs OpenMP
    * make Floyd_CPU.out
    * ./Floyd_CPU.out
  *To run CPU vs GPU
    * make Floyd_GPU.out
    * ./Floyd_GPU.out
