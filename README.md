# Ultralytics YOLO with Additional Knowledge Distillation Capability

<div align="center">
  <p>
    <a href="https://www.ultralytics.com/events/yolovision" target="_blank">
      <img width="100%" src="https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/banner-yolov8.png" alt="YOLO Vision banner">
      </a>
  </p>
</div>

## Core Idea

![image](https://github.com/user-attachments/assets/2acfbc67-7cf0-4591-ba27-cdf4370b88e2)


## How to Run

Example:

```
pip install -r requirements.txt
```

```
from ultralytics import YOLO

teacher_model = YOLO("<teacher-path>")

student_model = YOLO("yolo11n.pt)

student_model.train(
    data="<data-path>",
    teacher=teacher_model.model, # None if you don't wanna use knowledge distillation
    distillation_loss="logit",   # 'logit' (head-level, recommended), 'cwd' or 'mgd' (feature-level)
    epochs=100,
    batch=16,
    workers=0,
    exist_ok=True,
)
```

## Credits

- Channel-Wise Distillation Loss Implementation: [https://github.com/pppppM/mmdetection-distiller](https://github.com/pppppM/mmdetection-distiller)
- Mask Generation Distillation Loss Implementation: [https://github.com/yzd-v/MGD](https://github.com/yzd-v/MGD)
- YOLOv5 Knowledge Distillation Implementation: [https://github.com/wonbeomjang/yolov5-knowledge-distillation](https://github.com/wonbeomjang/yolov5-knowledge-distillation)

```
@software{Jocher_Ultralytics_YOLO_2023,
  author = {Jocher, Glenn and Qiu, Jing and Chaurasia, Ayush},
  license = {AGPL-3.0},
  month = jan,
  title = {{Ultralytics YOLO}},
  url = {https://github.com/ultralytics/ultralytics},
  version = {8.0.0},
  year = {2023}
}
```
