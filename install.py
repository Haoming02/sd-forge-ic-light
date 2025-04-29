import launch

if not launch.is_installed("onnxruntime"):
    launch.run_pip("install onnxruntime>=1.21.0", "onnxruntime for ic-light")

if not launch.is_installed("rembg"):
    launch.run_pip("install rembg==2.0.65", "rembg for ic-light")

if not launch.is_installed("cv2"):
    launch.run_pip("install opencv-python~=4.8.1", "opencv for ic-light")
