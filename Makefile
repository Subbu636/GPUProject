kmeans: KMain.cu KMeans.cu MetricSpace.cu
	nvcc -o kmeans KMain.cu KMeans.cu MetricSpace.cu
gmix: GMain.cu GMix.cu MetricSpace.cu
	nvcc -o gmix GMain.cu GMix.cu MetricSpace.cu
clean:
	$(RM) count *.o *~ kmeans gmix