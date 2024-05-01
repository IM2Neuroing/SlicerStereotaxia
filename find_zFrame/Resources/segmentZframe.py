import SimpleITK as sitk


def segment_zFrame(in_img: sitk.Image, img_type="MR", withPlots=False):
    import numpy as np

    from scipy.signal import butter, filtfilt

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: Failed to import matplotlib, not plots will be produced !")
        withPlots = False
    import math

    in_img_np = sitk.GetArrayFromImage(in_img)
    voxelCount = in_img_np.shape[0] * in_img_np.shape[1] * in_img_np.shape[2]

    hist_y, hist_x = np.histogram(in_img_np.flatten(), bins=int(in_img_np.max() / 4))

    hist_x = hist_x[1:]

    # filter the histogram

    cumHist_y = np.cumsum(hist_y.astype(float)) / voxelCount
    # we postulate that the background contain half of the voxels
    minThreshold_byCount = hist_x[np.where(cumHist_y > 0.90)[0][0]]
    print("background threshold by count found: %d" % minThreshold_byCount)

    # Mask in_image with minthreshold_byCount
    fg_img = in_img * sitk.Cast(in_img > minThreshold_byCount, in_img.GetPixelIDValue())
    # do edge detection on the fg_img
    edge_img = sitk.CannyEdgeDetection(sitk.Cast(fg_img, sitk.sitkFloat32))
    # merge labels that are in contact in the edge image
    connFilter = sitk.ConnectedComponentImageFilter()
    connFilter.SetFullyConnected(True)
    connected_img = connFilter.Execute(edge_img > 0)
    connected_img = sitk.RelabelComponent(connected_img, minimumObjectSize=100)
    # only keep connected components that have two sides larger than one third of the smallest
    # dimension of the image and the third that is 4 times smaller
    filtered_img = sitk.Image(in_img.GetSize(), sitk.sitkUInt8)
    filtered_img.CopyInformation(in_img)

    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.ComputeOrientedBoundingBoxOn()
    stats.Execute(connected_img)
    for label in stats.GetLabels():
        obb_size = np.array(stats.GetOrientedBoundingBoxSize(label))
        if np.product(obb_size) == 0:
            continue
        # compute the ratio between the smallest and the largest side of obb
        obb_ratio = max(obb_size) / min(obb_size)
        image_occupy = np.sum(obb_size >= 0.5 * min(in_img.GetSize()))
        if (image_occupy >= 1) and (obb_ratio > 4) and (np.sum(obb_size > 100) >= 2):
            print(f"Label {label} is a plate.")
            print(f"\tSize: {obb_size}")
            print(f"\tRatio: {obb_ratio}")
            filtered_img = sitk.Or(filtered_img, connected_img == label)
    # close holes in the resulting image
    closed_img = sitk.BinaryMorphologicalClosing(filtered_img, (1, 1, 1), sitk.sitkBall)
    return closed_img


def segment_zFrame_filesystem(in_file, out_file, img_type):
    import SimpleITK as sitk

    sitk.WriteImage(segment_zFrame(sitk.ReadImage(in_file), img_type, withPlots=True), out_file)


def segment_zFrame_slicer(inputVolume, outputVolume, img_type):
    import sitkUtils as siu

    try:
        import scipy
    except ImportError as e:
        slicer.util.pip_install("scipy")

    in_img = siu.PullVolumeFromSlicer(inputVolume)
    siu.PushVolumeToSlicer(segment_zFrame(in_img, img_type, withPlots=False), outputVolume)


if __name__ == "__main__":
    import argparse, os

    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        "-i", "--input_image", nargs=1, help="Stereotactic image.", required=True
    )
    argParser.add_argument(
        "-o", "--output_image", nargs=1, help="Segmented stereotactic frame.", required=True
    )
    argParser.add_argument("-t", "--image_type", nargs=1, help="MR/CT", required=False)
    args = argParser.parse_args()
    segment_zFrame_filesystem(
        os.path.abspath(args.input_image[0]),
        os.path.abspath(args.output_image[0]),
        args.image_type[0],
    )
