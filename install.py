import launch

if not launch.is_installed("rembg"):
    launch.run_pip("install rembg", "rembg")

for dep in ("onnxruntime", "pymatting", "pooch"):
    if not launch.is_installed(dep):
        launch.run_pip(f"install {dep}", f"{dep} for rembg")

if not launch.is_installed("pydantic"):
    launch.run_pip("install pydantic~=1.10.11", "pydantic for ic-light")

if not launch.is_installed("cv2"):
    launch.run_pip("install opencv-python~=4.8.0", "cv2 for ic-light")
