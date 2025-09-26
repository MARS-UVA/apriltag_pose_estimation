"""
This package is an executable package for calibrating a pinhole camera for 3D pose estimation from a video file.

To calibrate a camera, you will need a checkerboard pattern to calibrate against. The checkerboard should be on a flat
surface. If you're printing onto normal paper, you should tape the paper to a flat, rigid object. Some companies will
also sell higher-precision metal or acrylic calibration boards. calib.io provides a
`pattern generator <https://calib.io/pages/camera-calibration-pattern-generator>` on their website for generating
checkerboard patterns.

Calibrating a camera can be very sensitive, and many variables will affect your calibration result. You should
recalibrate your camera if you refocus it or replace its lens. Note that this means autofocusing cameras will likely
perform poorly at 3D pose estimation tasks.

In our experience, prerecording a video for calibrating a camera is faster than trying to calibrate in real time.
To get a good calibration video, try to get a variety of perspectives of the checkerboard (especially ones where the
checkerboard is heavily distorted from a square in the camera view from a variety of distances. You should also try to
cover the camera viewport entirely in your calibration video.

Not all frames from your video will be used for calibration. For example, if a frame is too blurry, the checkerboard
will not be detected and the frame will be skipped. The program will inform you how many frames it will process for
calibration. Usually above 50 is a reasonable number. Trying to calibrate with over 300 images is not recommended since
it will take a long time to compute. To allow for longer videos without taking too many frames, you may specify how many
frames are skipped before each frame evaluation.

If calibration is successful, the program will output a string which constructs a
:py:class:`~apriltag_pose_estimation.core.camera.CameraParameters` object representing the camera and a reprojection
error indicating how accurate the calibration is in units of pixels. To estimate the accuracy in normal distance units,
you would need the sensor size and the resolution of the captured image. For most cameras, a reprojection error under
1 px is acceptable. Be aware that the calibrator can overfit a calibration, which would cause a low reprojection error
despite being inaccurate. Use the tips provided above to avoid this problem.

========
Examples
========

The following shell command calibrates from a video file called ``video.mp4``, which features a checkerboard pattern of
8 squares by 11 squares, skipping 10 frames at a time::

   python3 -m apriltag_pose_estimation.calibrate video.mp4 --long 11 --short 8 --skip 10
"""
