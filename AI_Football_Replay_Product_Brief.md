# AI Football Replay --- Product Analysis Brief

## Vision

Build a web service that converts football video into an interactive 3D
reconstruction for replay and tactical analysis.

The product is **not** a video player. It is an AI system that
reconstructs the spatial state of the match and lets users inspect it
from arbitrary viewpoints.

------------------------------------------------------------------------

# Core Value Proposition

Instead of:

-   watching a fixed TV camera,

users can:

-   rotate the camera,
-   inspect the action from above,
-   analyze player movement,
-   inspect passing lanes,
-   analyze defensive shape,
-   create tactical explanations.

Goal:

> Turn football moments into interactive spatial experiences.

------------------------------------------------------------------------

# MVP

Input:

-   5--20 second football clip
-   single broadcast camera
-   preferably wide tactical view

Output:

-   3D pitch
-   tracked players
-   tracked ball
-   replay timeline
-   free camera
-   top-down tactical camera
-   trajectory visualization

No realistic body animation required.

------------------------------------------------------------------------

# Technology Stack

Frontend

-   Vite
-   TypeScript
-   Three.js r185

Backend

-   FastAPI
-   PostgreSQL
-   Redis
-   MinIO
-   FFmpeg

AI

-   PyTorch
-   YOLO
-   BoT-SORT
-   OpenCV
-   Camera calibration
-   Optional RTMPose later

------------------------------------------------------------------------

# Processing Pipeline

Video

↓

Shot detection

↓

Gameplay scene classification

↓

Player + ball detection

↓

Tracking

↓

Camera calibration

↓

World coordinates

↓

Trajectory smoothing

↓

Replay JSON

↓

Three.js renderer

------------------------------------------------------------------------

# First Target

Single continuous gameplay sequence.

NOT:

-   full match
-   montage
-   replay package
-   multiple cameras

------------------------------------------------------------------------

# Long-term Vision

Support:

-   full highlights
-   automatic segmentation
-   tactical reports
-   heatmaps
-   passing network
-   player identification
-   API

------------------------------------------------------------------------

# Product Positioning

Do NOT market as:

"Convert football video into 3D"

Instead:

"Explore football moments from any angle."

or

"AI reconstructs football moments into interactive tactical
simulations."

------------------------------------------------------------------------

# Commercial Strategy

Primary customers:

1.  Football academies
2.  Amateur clubs
3.  Coaches
4.  Analysts
5.  Football creators
6.  Sports media

World Cup serves primarily as a marketing event.

------------------------------------------------------------------------

# Growth Strategy

Publish one interactive reconstruction per important match.

Social teaser:

1.  Split screen:

    -   original clip
    -   synchronized 3D

2.  Transition to full 3D

3.  Rotate camera

4.  Tactical top-down view

5.  CTA: Explore the replay.

------------------------------------------------------------------------

# Legal Assumptions

Target architecture:

Video

↓

AI extraction

↓

Coordinates

↓

Delete source video

↓

Store only:

-   trajectories
-   camera parameters
-   metadata

Avoid storing or redistributing original footage.

------------------------------------------------------------------------

# Open Questions

## Technical

-   Camera calibration accuracy
-   Ball detection
-   Player identity
-   Automatic numbering
-   Broadcast shot filtering

## Product

-   Is 3D replay alone valuable?
-   Which features drive subscriptions?
-   Should editing be manual-assisted?

## Legal

-   Derivative work implications
-   User-uploaded broadcast footage
-   Tournament-specific restrictions
-   API licensing

------------------------------------------------------------------------

# Success Metrics

Technical

-   Stable tracking
-   Accurate field projection
-   Smooth replay
-   \<5 min processing per clip

Product

-   Users share replay links
-   Users revisit scenes
-   Coaches use during analysis

Business

-   Paid pilot customers
-   Monthly recurring revenue
-   API demand

------------------------------------------------------------------------

# Questions for an AI Strategy Agent

1.  Is this product differentiated enough?
2.  What is the strongest moat?
3.  What should be removed from MVP?
4.  What features create the most business value?
5.  What legal risks are highest?
6.  Which customer segment should be targeted first?
7.  Is there a stronger positioning than "3D replay"?
8.  What pricing model should be tested?
9.  How should the product evolve over three years?
10. What would make this venture venture-scale?
