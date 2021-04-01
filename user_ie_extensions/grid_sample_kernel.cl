
INPUT0_TYPE sample_input(__global INPUT0_TYPE* input, uint x, uint y, uint w, uint h)
{
    if (x < 0 || w <= x) return 0;
    if (y < 0 || h <= y) return 0;
    return input[y * INPUT0_PITCHES[2] + x * INPUT0_PITCHES[3]];
}

__kernel void grid_sample(
        __global INPUT0_TYPE* input,
        __global INPUT1_TYPE* grid,
        __global OUTPUT0_TYPE* out)
{
    const uint idx  = get_global_id(0);
    const uint idy  = get_global_id(1);
    const uint idbf = get_global_id(2);//batches*channels, as OpenCL supports 3D nd-ranges only
    const uint channel = idbf%OUTPUT0_DIMS[1];
    const uint batch   = idbf/OUTPUT0_DIMS[1];

    const uint inpHeight = INPUT0_DIMS[2];
    const uint inpWidth  = INPUT0_DIMS[3];

    //notice that pitches are in elements, not in bytes!
    const uint grid_offset  = INPUT1_OFFSET + (batch*INPUT1_PITCHES[0] + idy*INPUT1_PITCHES[1] + idx*INPUT1_PITCHES[2]);
    const uint out_offset   = OUTPUT0_OFFSET +    
        channel * OUTPUT0_PITCHES[1] + 
        1 * (batch*OUTPUT0_PITCHES[0] + idy*OUTPUT0_PITCHES[2]+ idx*OUTPUT0_PITCHES[3]);
 
    float input_x = 0.5f * (grid[grid_offset] + 1) * (inpWidth - 1);
    int x0 = floor(input_x);
    int x1 = x0 + 1;

    float input_y = 0.5f * (grid[grid_offset + 1] + 1) * (inpHeight - 1);
    int y0 = floor(input_y);
    int y1 = y0 + 1;

    __global INPUT0_TYPE* inp = input + batch * INPUT0_PITCHES[0] + channel * INPUT0_PITCHES[1] + INPUT0_OFFSET;
    INPUT0_TYPE v00 = sample_input(inp, x0, y0, inpWidth, inpHeight);
    INPUT0_TYPE v10 = sample_input(inp, x1, y0, inpWidth, inpHeight);
    INPUT0_TYPE v01 = sample_input(inp, x0, y1, inpWidth, inpHeight);
    INPUT0_TYPE v11 = sample_input(inp, x1, y1, inpWidth, inpHeight);

    out[out_offset] = v00 + 
        (input_y - y0) * (v01 - v00) +
        (input_x - x0) * (v10 - v00 +
        (input_y - y0) * (v11 - v10 - v01 + v00));
}

