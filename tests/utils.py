import SimpleITK as sitk
import subprocess

def np2nii(np_array):
    result_image = sitk.GetImageFromArray(np_array)
    return result_image


def run_cmd(cmd: str, name: str):
    process = subprocess.Popen(
        cmd.split(), stderr=subprocess.PIPE, stdout=subprocess.PIPE
    )
    stdout, stderr = process.communicate()
    exit_code = process.wait()
    assert exit_code == 0, f"{stderr}\nRun test {name} failed, please check"

