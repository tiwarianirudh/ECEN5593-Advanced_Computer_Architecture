


Floyd_CPU.out:Floyd_Warshall.c
	gcc $^ -fopenmp -o $@

Floyd_GPU.out:Floyd_Warshall.cu
	nvcc $^ -o $@

Dijkstra_CPU.out:Dijkstra.c
	gcc $^ -fopenmp -o $@

Dijkstra_GPU.out:Dijkstra.cu
	nvcc $^ -o $@

clean:
	rm -rf *.out
