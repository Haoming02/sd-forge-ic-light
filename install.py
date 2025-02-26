import launch

if not launch.is_installed("onnxruntime"):
    launch.run_pip(f"install onnxruntime", "onnxruntime for rembg")

if not launch.is_installed("rembg"):
    launch.run_pip("install rembg", "rembg")

if not launch.is_installed("cv2"):
    launch.run_pip("install opencv-python~=4.8.0", "cv2 for ic-light")
