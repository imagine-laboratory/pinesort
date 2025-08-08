<p align="center">
  <img src="assets/pipeline_pinesort.jpg?raw=true" width="99.1%" />
</p>

# PineSORT: A Simple Online Real-time Tracking Framework for Drone Videos in Agriculture

## 📝 Abstract

We introduce **PineSORT**, a novel Multiple Object Tracking (MOT) system for drone-based agricultural monitoring, specifically tracking pineapples for yield estimation. Our approach tackles key challenges such as repetitive patterns, similar object appearances, low frame rates, and drone motion effects. PineSORT enhances tracking accuracy with motion direction cost, camera motion compensation, a three-stage association strategy, and overlap management. To handle large displacements, we propose an ORB-based camera compensation technique that significantly improves Association Accuracy (AssA). Evaluated via 5-fold cross-validation against BoTSORT and AgriSORT, PineSORT achieves statistically significant gains in Identity-Switch Penalized IDF1 (ISP-IDF1), IDF1, HOTA, and AssA metrics, confirming its effectiveness for tracking low-FPS drone footage and its value for precision agriculture.

## 📁 Project Structure

```
src/
├── main.py                   # Main entry point to run PineSORT
├── requirements.txt          # Python dependencies
├── output/                   # Output folder for tracking results
│   └── track/
│       └── tracker/
│           ├── debug/        # Debugging info and logs
│           └── images/       # Visualizations of tracking outputs
├── output.txt                # Summary output file
└── tracker/
    ├── gmc.py                # Global motion compensation utilities
    ├── matching.py           # Object matching and association logic
    ├── pinesort.py           # Core PineSORT tracker implementation
    ├── track.py              # Tracking management code
    └── utils.py              # Helper functions
```

## 💻 Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/pinesort.git
   cd pinesort/src
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## ⚙️ Usage

Run PineSORT on a video dataset by specifying parameters as needed. Below are the available command-line arguments with their defaults and descriptions:

### Dataset and Evaluation Arguments

* `--path_video_imgs` (str): Path to the folder containing video frames as images.

### Detector

* `--model_path` (str): Path to YOLO object detector weights (Ultralytics YOLO).

### PineSORT Tracking Parameters

| Parameter                 | Type    | Default | Description                                                               |
| ------------------------- | ------- | ------- | ------------------------------------------------------------------------- |
| `--det_thresh`            | float   | 0.40    | Detection confidence threshold.                                           |
| `--min_det_thresh`        | float   | 0.30    | Minimum detection threshold to consider.                                  |
| `--max_age`               | int     | 5       | Max frames to keep alive a track without detection.                       |
| `--min_hits`              | int     | 1       | Min hits before confirming a track.                                       |
| `--first_iou_threshold`   | float   | 0.30    | First IOU threshold for association stage.                                |
| `--second_iou_threshold`  | float   | 0.30    | Second IOU threshold for association.                                     |
| `--third_iou_threshold`   | float   | 0.10    | Third IOU threshold for association.                                      |
| `--overlap_iou_threshold` | float   | 0.10    | IOU threshold to suppress redundant boxes using Kalman predictions.       |
| `--delta_t`               | int     | 3       | Time delta for motion model update.                                       |
| `--inertia`               | float   | 0.2     | Inertia coefficient for motion modeling.                                  |
| `--asso_func`             | str     | "eiou"  | Association function (e.g., "iou", "giou", "eiou", "ciou", "diou").       |
| `--use_byte`              | boolean | False   | Enable ByteTrack-style association logic (flag).                          |
| `--img_width`             | int     | 1920    | Image width in pixels.                                                    |
| `--img_height`            | int     | 1088    | Image height in pixels.                                                   |
| `--camera_compensation`   | str     | "orb"   | Camera compensation technique (e.g., "orb", "optisparseOptFlow", "sift"). |

### Tracking Arguments

| Parameter               | Type  | Default | Description                                           |
| ----------------------- | ----- | ------- | ----------------------------------------------------- |
| `--track_high_thresh`   | float | 0.3     | Confidence threshold for tracking.                    |
| `--track_low_thresh`    | float | 0.1     | Lowest detection threshold valid for tracks.          |
| `--new_track_thresh`    | float | 0.7     | Threshold to start a new track.                       |
| `--track_buffer`        | int   | 3       | Number of frames to keep lost tracks before deletion. |
| `--match_thresh`        | float | 0.7     | Matching threshold used during tracking association.  |
| `--aspect_ratio_thresh` | float | 1.6     | Filter out boxes with aspect ratio above this value.  |
| `--min_box_area`        | float | 10      | Filter out tiny bounding boxes below this area.       |

### Output Arguments

* `--output` (str): Directory to save tracking results (MOT format, debug info, images).
  *Default:* `./output/track/`
* `--output-debug` (flag): Save debug images with detector and Kalman predictions.
  *Default:* enabled (True)
* `--output-frames` (flag): Save images with tracking IDs overlaid.
  *Default:* enabled (True)

---

### Example command

```bash
python main.py \
  --path_video_imgs "./images" \
  --model_path "./best.pt" \
  --det_thresh 0.4 \
  --max_age 5 \
  --asso_func eiou \
  --camera_compensation orb \
  --output "./output/track/" \
  --output-debug \
  --output-frames
```


## 📖 Citation

If you find this repository useful, please star ⭐ the repository and cite:

```bibtex
@InProceedings{Xie-Li_2025_CVPR,
    author    = {Xie-Li, Danny and Fallas-Moya, Fabian},
    title     = {PineSORT: A Simple Online Real-time Tracking Framework for Drone Videos in Agriculture},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR) Workshops},
    month     = {June},
    year      = {2025},
    pages     = {65-74}
}
```

## Acknowledgements

We thank CeNAT (Centro Nacional de Alta Tecnología) for supporting this research through the CONARE scholarship program (Becas-CONARE), CNCA (Colaboratorio Nacional de Computación Avanzada) for development support, the University of Costa Rica for financial support via project C4612, and the Postgraduate Office of the Costa Rica Institute of Technology (ITCR) for publication assistance.