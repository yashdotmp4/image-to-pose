Problem Addressed:
Artists often rely on reference images to study anatomy, poses, and lighting, but it can be difficult to accurately interpret depth, proportion, and light direction from reference images exclusively. This project aims to provide a tool for artists to use to extract a pose from an image and apply it to a mannequin in a 3d space, along with the lighting detected from the image.

Existing Work:
Previous systems like OpenPose, VideoPose3D, and DeepMotion have focused on motion capture and 3D pose estimation for animation or VR. However, these are typically designed for performance tracking rather than artistic reference and rarely combine pose and lighting extraction.

Proposed Solution:
This project develops a deep learning–based system that takes a 2D image as input and outputs a 3D mannequin model with the pose and approximate lighting extracted from the image. It uses existing pose and lighting datasets and trains models from scratch to estimate joint positions and illumination direction.

Impact and Evaluation:
The system will be evaluated using pose accuracy metrics such as MPJPE and tested for visual realism in lighting reproduction. The impact lies in providing artists with a reference tool that simplifies pose study and improves understanding of form and light in digital art.
