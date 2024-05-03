import SimpleITK as sitk


def segment_zFrame(in_img: sitk.Image, img_type="MR", withPlots=False):
    import numpy as np

    from scipy.signal import butter, filtfilt

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: Failed to import matplotlib, not plots will be produced !")
        withPlots = False

    def isPlate_size(bbox: np.array) -> bool:
        """Check if the bounding box is a plate. The plate is defined as a large object
        that its largest side 0.3 of the smallest image side and has a ratio between the
        smallest and the largest side of the obb greater than 4.
        Args:
            bbox (np.array): a bounding box of an object as outputted by sitk.LabelShapeStatisticsImageFilter

        Returns:
            bool: if it is a plate
        """
        obb_size = np.array(bbox)
        if np.product(obb_size) == 0:
            return False
        # compute the ratio between the smallest and the largest side of obb
        obb_ratio = max(obb_size) / min(obb_size)
        image_occupy = np.sum(obb_size >= 0.5 * min(in_img_np.shape))
        return (image_occupy >= 1) and (obb_ratio > 4) and (np.sum(obb_size > 100) >= 2)

    def isPlate_loc(centroid, allObjects_bboxCenter, allObjects_bboxSize) -> bool:
        """Determine if the label is a plate based on its location in the image.
        Plates should have its centroid at least 30% of the smallest side of the
        image away from the center of the image

        Args:
            centroid (_type_): plate centroid
            allObjects_bboxCenter (_type_): centroid of all objects in image
            allObjects_bboxSize (_type_): bbox size of all objects in image

        Returns:
            bool: if it is a plate
        """
        return (
            np.linalg.norm(np.array(centroid) - np.array(allObjects_bboxCenter))
            > np.min(allObjects_bboxSize) * 0.3
        )

    def threshold_plates_CT(
        in_img: sitk.Image,
        minThreshold_byCount: float,
        hist_y: np.array,
        hist_x: np.array,
        withPlots=False,
    ) -> sitk.Image:
        """Threshold a CT image to segment the stereotactic plates.
        The threshold is the first maxima after the 90% threshold.

        Args:
            in_img (sitk.Image): The CT image
            minThreshold_byCount (float): threshold to isolate higher intensities (typically top 90%)
            hist_y (np.array): the histogram of the image
            hist_x (np.array): the histogram bins
            withPlots (bool, optional): wheither to plot graphs. Defaults to False.

        Returns:
            sitk.Image: The connected component image of the plates
        """
        b, a = butter(2, 0.15, btype="low", analog=False)
        hist_y = filtfilt(b, a, hist_y)

        hist_diff = np.diff(hist_y)

        hist_diff_zc = np.where(np.diff(np.sign(hist_diff)) == 2)[0].flatten()
        if withPlots:
            print(hist_x[hist_diff_zc])

        # minThreshold_byVariation = hist_x[hist_diff_zc[hist_x[hist_diff_zc]>secondSpikeVal_intens]][0]
        minThreshold_byVariation_id = hist_diff_zc[
            hist_x[hist_diff_zc] > (minThreshold_byCount + 100)
        ][0]
        minThreshold_byVariation = hist_x[minThreshold_byVariation_id]
        print("first maxima after soft tissue found: %f" % minThreshold_byVariation)

        if withPlots:
            fig, ax = plt.subplots(2, 1, sharex=True)
            ax[0].plot(hist_x, hist_y, label="intensity histogram")
            ax[1].plot(hist_x, cumHist_y, label="cumulative histogram")
            ax[1].axvline(x=minThreshold_byCount, ymin=0.0, ymax=1.0, color="r")

            ax[0].plot(hist_x[1:], hist_diff, label="derivative of intensity histogram")
            ax[0].plot(
                hist_x[hist_diff_zc],
                hist_diff[hist_diff_zc],
                linestyle="",
                marker="o",
                label="zero crossings of derivative of intensity histogram",
            )

            ax[0].axvline(x=minThreshold_byCount, ymin=0.0, ymax=1.0, color="r")
            ax[0].axvline(x=minThreshold_byVariation, ymin=0.0, ymax=1.0, color="g")
            fig.legend()
            ax[0].grid()

        middleSlice = np.array(np.floor(np.array(in_img_np.shape) / 2), dtype=int)

        thresh_img = in_img > minThreshold_byVariation

        # autocrop the thresholded volume to remove any background... (matlab...)
        # so that we can use relative sizes in `labelToKeep_mask`
        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(thresh_img)
        thresh_img = sitk.Crop(thresh_img, stats.GetBoundingBox(1))

        if withPlots:
            plt.figure()
            plt.imshow(sitk.GetArrayFromImage(thresh_img)[:, :, middleSlice[2]])

        connectedComponentImage = sitk.ConnectedComponent(thresh_img)
        return connectedComponentImage

    def threshold_plates_MR(in_img, minThreshold_byCount, withPlots=False) -> sitk.Image:
        """Threshold a MR image to segment the stereotactic plates.
        The plates are extracted using cannny edge detection and connected component analysis.

        Args:
            in_img (sitk.Image): The MR image
            minThreshold_byCount (float): threshold to isolate higher intensities (typically top 90%)
            withPlots (bool, optional): wheither to plot graphs. Defaults to False.

        Returns:
            sitk.Image: The connected component image of the plates
        """
        p90_img = in_img * sitk.Cast(in_img > minThreshold_byCount, in_img.GetPixelIDValue())
        # Mask in_image with minthreshold_byCount
        fg_img = p90_img * sitk.Cast(p90_img > minThreshold_byCount, p90_img.GetPixelIDValue())
        # do edge detection on the fg_img
        edge_img = sitk.CannyEdgeDetection(sitk.Cast(fg_img, sitk.sitkFloat32))
        # merge labels that are in contact in the edge image
        connFilter = sitk.ConnectedComponentImageFilter()
        connFilter.SetFullyConnected(True)
        connected_img = connFilter.Execute(edge_img > 0)
        connected_img = sitk.RelabelComponent(connected_img, minimumObjectSize=100)
        return connected_img

    def detectPlates(connected_img: sitk.Image) -> sitk.Image:
        """Shape analysis of the connected components to detect the plates

        Args:
            connected_img (sitk.Image): the connected component image after thresholding (each plate is a label)

        Returns:
            sitk.Image: connected image with only plates
        """
        # only keep connected components that have two sides larger than one third of the smallest
        # dimension of the image and the third that is 4 times smaller
        stats_allObjs = sitk.LabelShapeStatisticsImageFilter()
        stats_allObjs.ComputeOrientedBoundingBoxOn()
        stats_allObjs.Execute(connected_img > 0)
        crop_img = sitk.Crop(connected_img, stats_allObjs.GetBoundingBox(1))

        filtered_img = sitk.Image(crop_img.GetSize(), sitk.sitkUInt8)
        filtered_img.CopyInformation(crop_img)

        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.ComputeOrientedBoundingBoxOn()
        stats.Execute(crop_img)
        for label in stats.GetLabels():
            obb_size = np.array(stats.GetOrientedBoundingBoxSize(label))
            if np.product(obb_size) == 0:
                continue

            if isPlate_size(stats.GetOrientedBoundingBoxSize(label)) and isPlate_loc(
                stats.GetCentroid(label),
                stats_allObjs.GetCentroid(1),
                stats_allObjs.GetOrientedBoundingBoxSize(1),
            ):
                filtered_img = sitk.Or(filtered_img, crop_img == label)

        # close holes in the resulting image
        closed_img = sitk.BinaryMorphologicalClosing(filtered_img, (1, 1, 1), sitk.sitkBall)
        # paste the closed image on the original image space

        closed_img_orig = sitk.Image(connected_img.GetSize(), sitk.sitkUInt8)
        closed_img_orig.CopyInformation(connected_img)
        closed_img_orig = sitk.Paste(
            closed_img_orig,
            closed_img,
            closed_img.GetSize(),
            [0, 0, 0],
            closed_img_orig.TransformPhysicalPointToIndex(
                closed_img.TransformIndexToPhysicalPoint([0, 0, 0])
            ),
        )

        return closed_img_orig

    in_img_np = sitk.GetArrayFromImage(in_img)
    voxelCount = in_img_np.shape[0] * in_img_np.shape[1] * in_img_np.shape[2]

    hist_y, hist_x = np.histogram(in_img_np.flatten(), bins=int(in_img_np.max() / 4))

    hist_x = hist_x[1:]

    # filter the histogram

    cumHist_y = np.cumsum(hist_y.astype(float)) / voxelCount
    # we postulate that the background contain half of the voxels
    minThreshold_byCount = hist_x[np.where(cumHist_y > 0.90)[0][0]]

    print(f"background threshold by count found: {minThreshold_byCount}")

    if img_type == "CT":
        plates_img = threshold_plates_CT(in_img, minThreshold_byCount, hist_y, hist_x, withPlots)
    elif img_type == "MR":
        plates_img = threshold_plates_MR(in_img, minThreshold_byCount, withPlots)
    else:
        raise ValueError("Unknown image type")

    # paste the plate images on the original image space

    return detectPlates(plates_img)


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
