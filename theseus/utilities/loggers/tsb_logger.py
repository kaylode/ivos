from typing import Union
import io
import os
import glob
import torch
import traceback
import numpy as np
import pandas as pd
import PIL
from PIL import Image
from PIL.GifImagePlugin import Image as GifImage 
from torchvision.transforms import ToTensor
from torch.utils.tensorboard import SummaryWriter
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


from theseus.utilities.loggers.observer import LoggerObserver, LoggerSubscriber
LOGGER = LoggerObserver.getLogger('main')

class TensorboardLogger(LoggerSubscriber):
    """
    Logger for Tensorboard visualization
    :param log_dir: Path to save checkpoint
    """
    def __init__(self, log_dir, resume=None):
        self.log_dir = log_dir      
        self.writer = SummaryWriter(log_dir=self.log_dir)

        # Load old logging
        if resume is not None:
            self.load(resume)

    def log_scalar(self, tag, value, step, **kwargs):
        """
        Write a log to specified directory
        :param tags: (str) tag for log
        :param values: (number) value for corresponding tag
        :param step: (int) logging step
        """

        self.writer.add_scalar(tag, value, step)

    def log_figure(self, tag, value, step, **kwargs):
        """
        Write a matplotlib fig to tensorboard
        :param tags: (str) tag for log
        :param value: (image) image to log. torch.Tensor or plt.fire.Figure
        :param step: (int) logging step
        """

        if isinstance(value, torch.Tensor):
            self.writer.add_image(tag, value, global_step=step)
        else:
            self.writer.add_figure(tag, value, global_step=step)

    def log_torch_module(self, tag, value, inputs, **kwargs):
        """
        Write a model graph to tensorboard
        :param value: (nn.Module) torch model
        :param inputs: sample tensor
        """
        self.writer.add_graph(value, inputs)

    def log_embedding(self, tag, value, label_img=None, step=0, metadata=None, metadata_header=None, **kwargs):
        """
        Write a embedding projection to tensorboard
        :param value: (torch.Tensor) embedding (N, D)
        :param label_img: (torch.Tensor) normalized image tensors (N, 3, H, W)
        :param metadata: (List) list of coresponding labels
        """
        self.writer.add_embedding(
            tag=tag,
            mat=value, 
            label_img = label_img, 
            metadata=metadata, 
            metadata_header = metadata_header,
            global_step=step)


    def log_gif(self, tag, value, step, **kwargs):
        """
        Write a GIF summary of video
        :param value: (torch.Tensor) embedding (C, H, W, D)
        """

        summary = make_animated_gif_summary(
            tag = tag,
            image = value,
            frame_dim=-1,
            scale_factor=1.0
        )
        self.writer._get_file_writer().add_summary(summary, step)

    def load(self, old_log):
        """
        Load tensorboard from log
        :param old_log: (str) path to previous log
        """
        all_logs, all_figs = tflog2pandas(old_log)

        for _, row in all_logs.iterrows():
            tag, value, step = row
            self.log_scalar(tag,value,step)

        for _, row in all_figs.iterrows():
            tag, value, step = row
            image_result = Image.open(io.BytesIO(value))
            image = ToTensor()(image_result)
            self.log_figure(tag, image, step)

    def __del__(self):
        self.writer.close()


def tflog2pandas(path: str) -> pd.DataFrame:
    """convert single tensorflow log file to pandas DataFrame
    Parameters
    ----------
    path : str
        path to tensorflow log file
    Returns
    -------
    pd.DataFrame
        converted dataframe
    """
    DEFAULT_SIZE_GUIDANCE = {
        "compressedHistograms": 1,
        "images": 1,
        "scalars": 0,  # 0 means load all
        "histograms": 1,
    }
    runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})
    runfig_data = pd.DataFrame({"name": [], "value": [], "step": []})
    try:

        ## Scalar values
        event_acc = EventAccumulator(path, DEFAULT_SIZE_GUIDANCE)
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            r = {"metric": [tag] * len(step), "value": values, "step": step}
            r = pd.DataFrame(r)
            runlog_data = pd.concat([runlog_data, r])

        ## Image values
        tags = event_acc.Tags()["images"]
        for tag in tags:
            event_list = event_acc.Images(tag)
            values = list(map(lambda x: x.encoded_image_string, event_list))
            step = list(map(lambda x: x.step, event_list))

            r = {"name": [tag] * len(step), "value": values, "step": step}
            r = pd.DataFrame(r)
            runfig_data = pd.concat([runfig_data, r])

    # Dirty catch of DataLossError
    except Exception:
        LOGGER.text("Event file possibly corrupt: {}".format(path), level=LoggerObserver.WARN)
        traceback.print_exc()
    return runlog_data, runfig_data

def find_old_log(weight_path):
    """
    Find log inside dir
    """
    pardir = os.path.dirname(weight_path)
    event_paths = glob.glob(os.path.join(pardir, "event*"))
    if len(event_paths) == 0:
        return None
    else:
        return event_paths[0]





def _image3_animated_gif(
    tag: str, image: np.ndarray, frame_dim: int = 0, scale_factor: float = 1.0
):
    """Function to actually create the animated gif.
    Args:
        tag: Data identifier
        image: 3D image tensors expected to be in `HWD` format
        frame_dim: the dimension used as frames for GIF image, expect data shape as `HWD`, default to `0`.
        scale_factor: amount to multiply values by. if the image data is between 0 and 1, using 255 for this value will
            scale it to displayable range
    """
    if len(image.shape) != 3:
        raise AssertionError("3D image tensors expected to be in `HWD` format, len(image.shape) != 3")

    ims = [(i * scale_factor).astype(np.uint8, copy=False) for i in np.moveaxis(image, frame_dim, 0)]
    ims = [GifImage.fromarray(im) for im in ims]
    img_str = b""
    for b_data in PIL.GifImagePlugin.getheader(ims[0])[0]:
        img_str += b_data
    img_str += b"\x21\xFF\x0B\x4E\x45\x54\x53\x43\x41\x50" b"\x45\x32\x2E\x30\x03\x01\x00\x00\x00"
    for i in ims:
        for b_data in PIL.GifImagePlugin.getdata(i):
            img_str += b_data
    img_str += b"\x3B"

    summary_image_str = Summary.Image(height=10, width=10, colorspace=1, encoded_image_string=img_str)
    image_summary = Summary.Value(tag=tag, image=summary_image_str)
    return Summary(value=[image_summary])

def make_animated_gif_summary(
    tag: str,
    image: Union[np.ndarray, torch.Tensor],
    frame_dim: int = -3,
    scale_factor: float = 1.0,
) -> Summary:
    """Creates an animated gif out of an image tensor in 'CHWD' format and returns Summary.

    Args:
        tag: Data identifier
        image: The image, expected to be in `CHWD` format
        frame_dim: the dimension used as frames for GIF image, expect input data shape as `CHWD`,
            default to `-3` (the first spatial dim)
        scale_factor: amount to multiply values by.
            if the image data is between 0 and 1, using 255 for this value will scale it to displayable range
    """

    # GIF image has no channel dim, reduce the spatial dim index if positive
    frame_dim = frame_dim - 1 if frame_dim > 0 else frame_dim
    return _image3_animated_gif(tag, image, frame_dim, scale_factor)

    # summary_op = []
    #     one_channel_img: Union[torch.Tensor, np.ndarray] = (
    #         image[it_i, :, :, :].squeeze(dim=0) if isinstance(image, torch.Tensor) else image[it_i, :, :, :]
    #     )
       
    # return summary_op