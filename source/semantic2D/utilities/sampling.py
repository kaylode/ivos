import numpy as np

def sampling_frames(num_frames, max_frames=-1, uniform=False, sampling_rate=0):
    """
    num_frames: `int`
        number of frames to sample from
    max_frames: `int`, `float`
        maximum number of frames to return
    uniform: `bool`
        whether to perform uniform sampling
    sampling_rate: `int`
        if uniform sampling, number of steps between frames is required
    """
    if max_frames == -1:
        max_frames = num_frames
    
    # Max frames can be int or float
    if isinstance(max_frames, float):
        _max_frames = int(max_frames * num_frames)
    else:
        _max_frames = max_frames

    if uniform:
        assert sampling_rate > 0, "To do uniform sampling, sampling rate must be greater than 0"
        # Pre-uniform sampling
        uniform_num_frames = num_frames // sampling_rate # number of frames based on rate
        current_sample_indx = np.linspace(
            0, num_frames - 1, 
            num=uniform_num_frames, dtype=int) # sample frames, with step equals sample_fp
        
        # if the length of current_sample_indx is already less than max_frames, just use the current version to tensor
        # else continue to uniformly sample the frames whose length is max_frames
        # when training, the frames are sampled randomly in the uniform split interval
        if _max_frames >=  current_sample_indx.shape[0]:
            frame_index = np.arange(0, current_sample_indx.shape[0])
        else:
            frame_index = np.linspace(0, current_sample_indx.shape[0] - 1, num=_max_frames, dtype=int)

        sampled_frame_ids = [current_sample_indx[int(index)] for index in frame_index]
    else:
        # If num frames is fewer than max_frames, use all
        if num_frames < _max_frames:
            return [i for i in range(num_frames)]

        # Random sampling
        sampled_frame_ids = np.random.choice(range(num_frames), size=_max_frames, replace=False)
        sampled_frame_ids = sampled_frame_ids.tolist()

    # Sort frames by id
    sampled_frame_ids.sort()

    return sampled_frame_ids