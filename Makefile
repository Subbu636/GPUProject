kmeans: KMain.cu KMeans.cu MetricSpace.cu
	nvcc KMain.cu KMeans.cu MetricSpace.cu -o kmeans
gmix: GMain.cu GMix.cu MetricSpace.cu
	nvcc -lcublas GMain.cu GMix.cu MetricSpace.cu -o gmix 
gmix-old: GMain.cu GMix-old.cu MetricSpace.cu
	nvcc -lcublas  GMain.cu GMix-old.cu MetricSpace.cu -o gmix-old 
gmix-cuBlas: GMain.cu GMix-cuBlas.cu MetricSpace.cu
	nvcc -lcublas  GMain.cu GMix-cuBlas.cu MetricSpace.cu -o gmix-cublas
gmix-better: GMain.cu GMix.cu MetricSpace.cu
	nvcc -lcublas -arch=sm_80 GMain.cu GMix.cu MetricSpace.cu -o gmix-better
clean:
	$(RM) count *.o *~ kmeans gmix