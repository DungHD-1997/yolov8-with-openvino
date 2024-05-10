import ipywidgets as widgets
import openvino as ov

from PIL import Image
from ultralytics import YOLO

models_dir = "C:/Project/result_train/yolov8/clean_data_obj/weights/best.pt"
det_model = YOLO(models_dir)
label_map = det_model.model.names

IMAGE_PATH = "C:/Project/test_img_truck/image_2023-12-25_07z37z31.jpg"
res = det_model(IMAGE_PATH)
# Image.fromarray(res[0].plot()[:, :, ::-1])
for i, r in enumerate(res):
    # Plot results image
    im_bgr = r.plot()  # BGR-order numpy array
    im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

    # Show results to screen (in supported environments)
    r.show()

    # Show results to screen (in supported environments)
    # res.show()

# core = ov.Core()

# device = widgets.Dropdown(
#     options=core.available_devices + ["AUTO"],
#     value="AUTO",
#     description="Device:",
#     disabled=False,
# )

# import torch

# core = ov.Core()
# det_model_path = "C:/Project/result_train/yolov8/clean_data_obj/weights/best_openvino_model/best.xml"
# det_ov_model = core.read_model(det_model_path)

# ov_config = {}
# if device.value != "CPU":
#     det_ov_model.reshape({0: [1, 3, 640, 640]})
# if "GPU" in device.value or ("AUTO" in device.value and "GPU" in core.available_devices):
#     ov_config = {"GPU_DISABLE_WINOGRAD_CONVOLUTION": "YES"}
# det_compiled_model = core.compile_model(det_ov_model, device.value, ov_config)


# def infer(*args):
#     result = det_compiled_model(args)
#     return torch.from_numpy(result[0])


# det_model.predictor.infer = infer
# det_model.predict.model.pt = False

# res = det_model(IMAGE_PATH)
# # Image.fromarray(res[0].plot()[:, :, ::-1])
# for i, r in enumerate(res):
#     # Plot results image
#     im_bgr = r.plot()  # BGR-order numpy array
#     im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

#     # Show results to screen (in supported environments)
#     r.show()