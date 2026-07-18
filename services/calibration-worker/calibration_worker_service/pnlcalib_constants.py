INPUT_WIDTH = 960
INPUT_HEIGHT = 540
KEYPOINT_THRESHOLD = 0.1611
LINE_THRESHOLD = 0.3434
CACHE_SCHEMA_VERSION = "pnlcalib-points-lines-v2"


# Official PnLCalib line-channel order. Goal-frame segments remain useful
# visual evidence but are not coplanar with the grass homography.
SEMANTIC_LINE_NAMES = (
    "Big rect. left bottom",
    "Big rect. left main",
    "Big rect. left top",
    "Big rect. right bottom",
    "Big rect. right main",
    "Big rect. right top",
    "Goal left crossbar",
    "Goal left post left",
    "Goal left post right",
    "Goal right crossbar",
    "Goal right post left",
    "Goal right post right",
    "Middle line",
    "Side line bottom",
    "Side line left",
    "Side line right",
    "Side line top",
    "Small rect. left bottom",
    "Small rect. left main",
    "Small rect. left top",
    "Small rect. right bottom",
    "Small rect. right main",
    "Small rect. right top",
)
GOAL_FRAME_LINE_IDS = frozenset(range(7, 13))
