import SimpleITK as sitk
import logging

logging.basicConfig(level=logging.DEBUG)


def segment_zFrame(in_img: sitk.Image, img_type="MR", withPlots=False):
    import numpy as np
    import logging
    from scipy.signal import butter, filtfilt

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logging.debug(
            "WARNING: Failed to import matplotlib, not plots will be produced !"
        )
        withPlots = False

    # first, upsample in_image to isotropic resolution
    # in_img_iso = sitk.Resample(
    #     in_img,
    #     in_img.GetSize(),
    #     sitk.Transform(3, sitk.sitkIdentity),
    #     sitk.sitkBSpline,
    #     in_img.GetOrigin(),
    #     [np.min(in_img.GetSpacing())] * 3,
    #     in_img.GetDirection(),
    #     0,
    #     in_img.GetPixelIDValue(),
    # )
    in_img_iso = in_img

    def calculate_surface_distances(reference_segmentation, segmented_segmentation):
        """
        Calculate the surface distances between two segmentations.

        Args:
            reference_segmentation (SimpleITK.Image): The reference segmentation.
            segmented_segmentation (SimpleITK.Image): The segmented segmentation.

        Returns:
            list: A list of surface distances between the two segmentations.
        """
        # Use the absolute values of the distance map to compute the surface distances (distance map sign, outside or inside
        # relationship, is irrelevant)
        reference_distance_map = sitk.Abs(
            sitk.SignedMaurerDistanceMap(
                reference_segmentation, squaredDistance=False, useImageSpacing=True
            )
        )
        reference_surface = sitk.LabelContour(reference_segmentation)

        segmented_distance_map = sitk.Abs(
            sitk.SignedMaurerDistanceMap(
                segmented_segmentation, squaredDistance=False, useImageSpacing=True
            )
        )
        segmented_surface = sitk.LabelContour(segmented_segmentation)

        statistics_image_filter = sitk.StatisticsImageFilter()
        # Get the number of pixels in the reference surface by counting all pixels that are 1.
        statistics_image_filter.Execute(reference_surface)
        num_reference_surface_pixels = int(statistics_image_filter.GetSum())

        # Get the number of pixels in the segmented surface by counting all pixels that are 1.
        statistics_image_filter.Execute(segmented_surface)
        num_segmented_surface_pixels = int(statistics_image_filter.GetSum())

        # Multiply the binary surface segmentations with the distance maps. The resulting distance
        # maps contain non-zero values only on the surface (they can also contain zero on the surface)
        seg2ref_distance_map = reference_distance_map * sitk.Cast(
            segmented_surface, sitk.sitkFloat32
        )
        ref2seg_distance_map = segmented_distance_map * sitk.Cast(
            reference_surface, sitk.sitkFloat32
        )

        # Get all non-zero distances and then add zero distances if required.
        seg2ref_distance_map_arr = sitk.GetArrayViewFromImage(seg2ref_distance_map)
        seg2ref_distances = list(
            seg2ref_distance_map_arr[seg2ref_distance_map_arr != 0]
        )
        seg2ref_distances = seg2ref_distances + list(
            np.zeros(num_segmented_surface_pixels - len(seg2ref_distances))
        )

        ref2seg_distance_map_arr = sitk.GetArrayViewFromImage(ref2seg_distance_map)
        ref2seg_distances = list(
            ref2seg_distance_map_arr[ref2seg_distance_map_arr != 0]
        )
        ref2seg_distances = ref2seg_distances + list(
            np.zeros(num_reference_surface_pixels - len(ref2seg_distances))
        )

        all_surface_distances = seg2ref_distances + ref2seg_distances

        return all_surface_distances

    def pasteImageInRef(in_img: sitk.Image, ref_img: sitk.Image) -> sitk.Image:
        out_img = sitk.Image(ref_img.GetSize(), ref_img.GetPixelIDValue())
        out_img.CopyInformation(ref_img)
        out_img = sitk.Paste(
            ref_img,
            sitk.Cast(in_img, ref_img.GetPixelIDValue()),
            in_img.GetSize(),
            [0, 0, 0],
            ref_img.TransformPhysicalPointToIndex(
                in_img.TransformIndexToPhysicalPoint([0, 0, 0])
            ),
        )
        return out_img

    def isPlate_size(bbox: np.array, in_img: sitk.Image) -> bool:
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
        img_extend = np.array(in_img.GetSize()) * np.array(in_img.GetSpacing())
        image_occupy = np.sum(obb_size >= (0.5 * min(img_extend)))
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
            logging.debug(hist_x[hist_diff_zc])

        # minThreshold_byVariation = hist_x[hist_diff_zc[hist_x[hist_diff_zc]>secondSpikeVal_intens]][0]
        minThreshold_byVariation_id = hist_diff_zc[
            hist_x[hist_diff_zc] > (minThreshold_byCount + 100)
        ][0]
        minThreshold_byVariation = hist_x[minThreshold_byVariation_id]
        logging.debug(
            "first maxima after soft tissue found: %f" % minThreshold_byVariation
        )

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

    def threshold_plates_MR(
        in_img, minThreshold_byCount, withPlots=False
    ) -> sitk.Image:
        """Threshold a MR image to segment the stereotactic plates.
        The plates are extracted using cannny edge detection and connected component analysis.

        Args:
            in_img (sitk.Image): The MR image
            minThreshold_byCount (float): threshold to isolate higher intensities (typically top 90%)
            withPlots (bool, optional): wheither to plot graphs. Defaults to False.

        Returns:
            sitk.Image: The connected component image of the plates
        """
        p90_img = in_img * sitk.Cast(
            in_img > minThreshold_byCount, in_img.GetPixelIDValue()
        )
        # Mask in_image with minthreshold_byCount
        fg_img = p90_img * sitk.Cast(
            p90_img > minThreshold_byCount, p90_img.GetPixelIDValue()
        )
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

            if isPlate_size(
                stats.GetOrientedBoundingBoxSize(label), crop_img
            ) and isPlate_loc(
                stats.GetCentroid(label),
                stats_allObjs.GetCentroid(1),
                stats_allObjs.GetOrientedBoundingBoxSize(1),
            ):
                logging.debug(f"Label {label} detected as plate")
                filtered_img = sitk.Or(filtered_img, crop_img == label)

        # close holes in the resulting image
        closed_img = sitk.BinaryMorphologicalClosing(
            filtered_img > 0, (1, 1, 1), sitk.sitkBall
        )
        # paste the closed image on the original image space
        closed_img_orig = pasteImageInRef(
            closed_img, sitk.Cast(connected_img * 0, closed_img.GetPixelIDValue())
        )

        return closed_img_orig

    def refinePlate_thres(plate_img: sitk.Image) -> sitk.Image:
        """Refine a plate image by optimizing a threshold starting from the max intensity of the plate
        down until:
        - the plate is a single connected component
        - the voxels in the thresholded image are no more than 1mm from the voxels in the original image (distance map)

        Args:
            plate_img (sitk.Image): a binary image of a single plate

        Raises:
            ValueError: _description_

        Returns:
            sitk.Image: A binary image of the same single plate but refined
        """
        import scipy.optimize as opt

        max_intensity = sitk.GetArrayFromImage(plate_img).max()
        min_intensity = sitk.GetArrayFromImage(plate_img).min()

        def objective(threshold):
            thresholded_img = plate_img > threshold
            largest_plate = plates_img > 0
            # extract contours from both images
            test_contour = sitk.LabelContour(thresholded_img)
            ref_contour = sitk.LabelContour(largest_plate)
            # compute the hausdorff distance between the two contours
            distance_map = sitk.HausdorffDistanceImageFilter()
            distance_map.Execute(test_contour, ref_contour)
            mean_distance = distance_map.GetAverageHausdorffDistance()
            # count the number of connected components
            connected_img = sitk.ConnectedComponent(thresholded_img)
            num_labels = sitk.GetArrayFromImage(connected_img).max() - 1
            if num_labels == 0:
                return 1000

            logging.debug(
                f"threshold: {threshold}, num_labels: {num_labels}, max_distance: {mean_distance}"
            )
            return num_labels / mean_distance

        result = opt.minimize_scalar(
            objective, bounds=(min_intensity, max_intensity), method="bounded"
        )
        refined_threshold = result.x
        refined_img = plate_img > refined_threshold

        return refined_img

    def refinePlate_seedGrow(plate_img: sitk.Image) -> sitk.Image:
        """Alternate version of refinePlate using a seed growing algorithm

        Args:
            plate_img (sitk.Image): _description_

        Returns:
            sitk.Image: _description_
        """
        import scipy.optimize as opt

        # Get the top 10% intensity voxels as seeds
        max_intensity = float(sitk.GetArrayFromImage(plate_img).max())
        min_intensity = float(sitk.GetArrayFromImage(plate_img).min())
        top_10_percent = np.percentile(
            sitk.GetArrayFromImage(plate_img)[sitk.GetArrayFromImage(plate_img) > 0], 95
        )
        seeds = np.reshape(
            np.argwhere(sitk.GetArrayFromImage(plate_img) >= top_10_percent), [-1, 3]
        )[:, ::-1].tolist()
        logging.debug(
            f"10 percentile threshold {top_10_percent} | Number of seeds: {len(seeds)}"
        )
        largest_plate = plates_img > 0
        ref_contour = sitk.LabelContour(largest_plate, fullyConnected=True)

        def objective(lower_threshold):
            # Perform seed growing with the given lower threshold
            thresholded_img = sitk.ConnectedThreshold(
                plate_img, seedList=seeds, lower=lower_threshold, upper=max_intensity
            )
            thres_img_conn = sitk.ConnectedComponent(thresholded_img)
            label_number = sitk.GetArrayFromImage(thres_img_conn).max()
            # extract contours from both images
            test_contour = sitk.LabelContour(thresholded_img, fullyConnected=True)
            mean_distance = np.mean(
                calculate_surface_distances(test_contour, ref_contour)
            )
            logging.debug(
                f"Lower threshold: {lower_threshold} \n\tMean distance: {mean_distance}"
            )
            return (10 * label_number) / mean_distance

        # Optimize the lower threshold to minimize the mean Hausdorff distance
        logging.debug(
            f"Optimizing lower threshold between {min_intensity} and {0.9*top_10_percent}"
        )
        result = opt.minimize_scalar(
            objective, bounds=(min_intensity, 0.9 * top_10_percent), method="bounded"
        )
        refined_lower_threshold = result.x

        # Perform seed growing with the refined lower threshold
        refined_seg_img = sitk.ConnectedThreshold(
            plate_img,
            seedList=seeds,
            lower=refined_lower_threshold,
            upper=max_intensity,
        )

        return refined_seg_img

    def refinePlate_squeletize_itk(plate_img: sitk.Image) -> sitk.Image:
        """Squeletize the plate

        Args:
            plate_img (sitk.Image): The plate image

        Returns:
            sitk.Image: The plate image as 1 voxel thick
        """
        binary_thinning = sitk.BinaryThinningImageFilter()
        skeleton_img = binary_thinning.Execute(plate_img)
        # This is not very good, as it is based on a 2D implementation,
        # it thins a lot in one axis
        # what comes after does not work well either: the BinaryThinning filter produces a very flat skeleton
        # (thin in only one direction) then using the values in the result of the thinning to do the connected threshold
        # does not change much, as we include voxels with low values as well.
        # seeds = sitk.GetArrayFromImage(skeleton_img) > 0
        # seedsLoc = np.argwhere(seeds)
        # seedsLoc_flat = np.reshape(seedsLoc, [-1, 3])
        # seeds_values = sitk.GetArrayFromImage(plate_img)[seeds]
        # print(
        #     f"Min seed value {np.min(seeds_values)} | Max seed value {np.max(seeds_values)}"
        # )
        # thresholded_img = sitk.ConnectedThreshold(
        #     plate_img,
        #     seedList=seedsLoc_flat[:, ::-1].tolist(),
        #     lower=float(np.min(seeds_values)),
        #     upper=float(np.max(seeds_values)),
        # )

        return skeleton_img

    def refined_plate_squeletize_skimage(plate_img: sitk.Image) -> sitk.Image:
        """Squeletize the plate

        Args:
            plate_img (sitk.Image): The plate image

        Returns:
            sitk.Image: The plate image as 1 voxel thick
        """
        import skimage.morphology as skmorph

        plate_img_np = sitk.GetArrayFromImage(plate_img)
        plate_img_np = plate_img_np > 0
        plate_img_np = skmorph.skeletonize(plate_img_np)
        res = sitk.GetImageFromArray(plate_img_np.astype(np.uint8))
        res.SetDirection(plate_img.GetDirection())
        res.SetOrigin(plate_img.GetOrigin())
        res.SetSpacing(plate_img.GetSpacing())
        
        return sitk.BinaryDilate(res, (1,1,1), sitk.sitkBall)

    def refinePlates(plates_img: sitk.Image, refineMethod: str) -> sitk.Image:
        """Refine all plates in the image

        Args:
            plates_img (sitk.Image): the plate image masked with the plate mask
            refineMethod (str): the method to refine the plates ("thres", "seedGrow" or "squeletize")

        Returns:
            sitk.Image: a binary image of all plates refined
        """
        refined_img = sitk.Image(plates_img.GetSize(), plates_img.GetPixelIDValue())
        refined_img.CopyInformation(plates_img)
        plates_conn = sitk.ConnectedComponent(plates_img)
        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(plates_conn)
        if stats.GetNumberOfLabels() > 10:
            logging.debug(
                f"Number of plates detected: {stats.GetNumberOfLabels()}. Something went wrong with the segmentation."
            )
            return ValueError("Too many plates detected")
        for label in stats.GetLabels():
            logging.debug(f"Refining plate {label}")
            thisPlate_mask = plates_conn == label
            thisPlate_img = sitk.Mask(plates_img, thisPlate_mask)

            if refineMethod == "thres":
                refined_plate = refinePlate_thres(thisPlate_img)
            elif refineMethod == "seedGrow":
                refined_plate = refinePlate_seedGrow(thisPlate_img)
            elif refineMethod == "squeletize_itk":
                refined_plate = refinePlate_squeletize_itk(thisPlate_img)
            elif refineMethod == "squeletize_skimage":
                refined_plate = refined_plate_squeletize_skimage(thisPlate_img)
            else:
                raise ValueError("Unknown refinement method")

            refined_img = sitk.Or(
                refined_img, sitk.Cast(refined_plate, refined_img.GetPixelIDValue())
            )
        return refined_img

    ################################################################################################
    ## MAIN
    ################################################################################################
    in_img_np = sitk.GetArrayFromImage(in_img_iso)
    voxelCount = in_img_np.shape[0] * in_img_np.shape[1] * in_img_np.shape[2]

    hist_y, hist_x = np.histogram(in_img_np.flatten(), bins=int(in_img_np.max() / 4))
    hist_x = hist_x[1:]

    # filter the histogram
    cumHist_y = np.cumsum(hist_y.astype(float)) / voxelCount
    # we postulate that the background contain half of the voxels
    minThreshold_byCount = hist_x[np.where(cumHist_y > 0.90)[0][0]]
    logging.debug(f"background threshold by count found: {minThreshold_byCount}")

    if img_type == "CT":
        plates_img = threshold_plates_CT(
            in_img_iso, minThreshold_byCount, hist_y, hist_x, withPlots
        )
    elif img_type == "MR":
        plates_img = threshold_plates_MR(in_img_iso, minThreshold_byCount, withPlots)
    else:
        raise ValueError("Unknown image type")
    # paste the plate images on the original image space
    refined_plates = refinePlates(detectPlates(plates_img) > 0, "squeletize_skimage")
    # refined_plates = detectPlates(plates_img)>0
    return (
        sitk.Mask(
            in_img_iso,
            pasteImageInRef(
                refined_plates > 0,
                sitk.Cast(in_img_iso * 0, refined_plates.GetPixelIDValue()),
            ),
        )
        > 0
    )


def segment_zFrame_filesystem(in_file, out_file, img_type):
    import SimpleITK as sitk

    sitk.WriteImage(
        segment_zFrame(sitk.ReadImage(in_file), img_type, withPlots=True), out_file
    )


def segment_zFrame_slicer(inputVolume, outputVolume, img_type):
    import sitkUtils as siu

    try:
        import scipy, skimage
    except ImportError as e:
        slicer.util.pip_install("scipy")
        slicer.util.pip_install("scikit-image")
        import scipy, skimage

    in_img = siu.PullVolumeFromSlicer(inputVolume)
    siu.PushVolumeToSlicer(
        segment_zFrame(in_img, img_type, withPlots=False), outputVolume
    )


if __name__ == "__main__":
    import argparse, os
    import logging
    import numpy as np
    import SimpleITK as sitk

    logging.basicConfig(level=logging.DEBUG)
    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        "-i", "--input_image", nargs=1, help="Stereotactic image.", required=True
    )
    argParser.add_argument(
        "-o",
        "--output_image",
        nargs=1,
        help="Segmented stereotactic frame.",
        required=True,
    )
    argParser.add_argument("-t", "--image_type", nargs=1, help="MR/CT", required=False)
    args = argParser.parse_args()
    segment_zFrame_filesystem(
        os.path.abspath(args.input_image[0]),
        os.path.abspath(args.output_image[0]),
        args.image_type[0],
    )
