struct Color {
   unsigned char R;
   unsigned char G;
   unsigned char B;
};  

/*
    Assignes pixel to closest cluster
*/

__kernel void assignToCluster(__global unsigned char *imageIn, 
                        __global int *c, 
                        __global struct Color *centroids, 
                        __global int *clusterCount,
                        int height,
                        int width
                        ) {    
    int locID = get_local_id(0);
    int globID = get_global_id(0);

    if (globID < height * width) {

        __local struct Color local_centroids[K];
        __local int local_clusterCount[K*4];


        if (locID < K) {
            local_centroids[locID].R = centroids[locID].R;    
            local_centroids[locID].G = centroids[locID].G;
            local_centroids[locID].B = centroids[locID].B;

            local_clusterCount[locID*4] = 0;
            local_clusterCount[locID*4+1] = 0;
            local_clusterCount[locID*4+2] = 0;
            local_clusterCount[locID*4+3] = 0;
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);

        struct Color pixel = { 
            .R = imageIn[globID*4+2], 
            .G = imageIn[globID*4+1], 
            .B = imageIn[globID*4] 
        };


        double minDist = DBL_MAX;
        int minIndex = 0.0;

        // Assign pixel to closest cluster        
        for (int i = 0; i < K; i++) {
            // Calculate distance 
            double dB = local_centroids[i].B - pixel.B;
            double dG = local_centroids[i].G - pixel.G;
            double dR = local_centroids[i].R - pixel.R;
            
            double dist = dB * dB + dG * dG + dR * dR;

            if(dist < minDist) {
                minIndex = i;
                minDist = dist;
            }
        }


        atomic_add(&local_clusterCount[4*minIndex], pixel.R);
        atomic_add(&local_clusterCount[4*minIndex+1], pixel.G);
        atomic_add(&local_clusterCount[4*minIndex+2], pixel.B);
        atomic_inc(&local_clusterCount[4*minIndex+3]);

        barrier(CLK_LOCAL_MEM_FENCE);

        if (locID < K) {
            atomic_add(&clusterCount[4*locID], local_clusterCount[4*locID]);
            atomic_add(&clusterCount[4*locID+1], local_clusterCount[4*locID+1]);
            atomic_add(&clusterCount[4*locID+2], local_clusterCount[4*locID+2]);
            atomic_add(&clusterCount[4*locID+3], local_clusterCount[4*locID+3]); 
        }

        c[globID] = minIndex;
    }
}



/*
    Updates clusters (centroid positions)
*/

__kernel void updateCentroids(__global struct Color *centroids, 
                            __global int *clusterCount, 
                            __global int *c, 
                            __global int *randIndexes,
                            __global unsigned char *imageIn
                            ) {
    int globID = get_global_id(0);

    if (globID < K) {    
        int count = clusterCount[4*globID+3];
        
        if (count == 0) {
            // Fix empty cluster
            int randIndex = randIndexes[globID];
            c[randIndex] = globID;

            atomic_add(&clusterCount[4*globID], imageIn[randIndex*4+2]);
            atomic_add(&clusterCount[4*globID+1], imageIn[randIndex*4+1]);
            atomic_add(&clusterCount[4*globID+2], imageIn[randIndex*4]);
            atomic_inc(&clusterCount[4*globID+3]);
            
            count = 1;
        }
        centroids[globID].B = clusterCount[4*globID+2] / count;
        centroids[globID].G = clusterCount[4*globID+1] / count; 
        centroids[globID].R = clusterCount[4*globID] / count;         
    }    
}
