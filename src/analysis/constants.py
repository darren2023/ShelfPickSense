"""与 visual-dps-data-collector 对齐的常量。"""

COCO17_KEYPOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

LEFT_WRIST_IDX = 9
RIGHT_WRIST_IDX = 10
LEFT_SHOULDER_IDX = 5
RIGHT_SHOULDER_IDX = 6

SKELETON_FILE = "skeleton.parquet"
ANNOTATION_FILE = "annotation.json"
EVENT_REVIEW_FILE = "event_review.json"
