import SimpleITK as sitk


def load_ct_info(file_path):
    sitk_image = sitk.ReadImage(file_path)
    if sitk_image is None:
        res = {}
    else:
        origin = list(reversed(sitk_image.GetOrigin()))
        spacing = list(reversed(sitk_image.GetSpacing()))
        direction = sitk_image.GetDirection()
        direction = [direction[8], direction[4], direction[0]]
        res = {"sitk_image": sitk_image,
               "npy_image": sitk.GetArrayFromImage(sitk_image),
               "origin": origin,
               "spacing": spacing,
               "direction": direction}
    return res

def save_ct_from_sitk(sitk_image, save_path, sitk_type=None, use_compression=False):
    if sitk_type is not None:
        sitk_image = sitk.Cast(sitk_image, sitk_type)
    sitk.WriteImage(sitk_image, save_path, use_compression)

def save_ct_from_npy(npy_image, save_path, origin=None, spacing=None,
                     direction=None, sitk_type=None, use_compression=False):
    sitk_image = sitk.GetImageFromArray(npy_image)
    if origin is not None:
        sitk_image.SetOrigin(origin)
    if spacing is not None:
        sitk_image.SetSpacing(spacing)
    if direction is not None:
        sitk_image.SetDirection(direction)
    save_ct_from_sitk(sitk_image, save_path, sitk_type, use_compression)