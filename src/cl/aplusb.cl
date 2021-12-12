#ifdef __CLION_IDE__
    #include "clion_defines.cl"
#endif

#line 6
__kernel void aplusb(__global const float* a,
                     __global const float* b,
                     __global float* c,
                     unsigned const int n) {
    const size_t id = get_global_id(0);
    if (id < n)
        c[id] = a[id] + b[id];
}
