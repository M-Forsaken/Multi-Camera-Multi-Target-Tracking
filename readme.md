## **Description**

Multi-camera Multi-target tracking system for non-overlapping camera setups. Using Yolo/Yolox with OSNet re-id.

The system makes use of:
- ByteTrack algorithm combined with KLT Tracker.
- Yolox/Yolo detector

## Installation

```python
pip install -r requirements.txt
```
## How to Run
```bash
python app.py
```


## Acknowledgement

This project benefits greatly from the works of
[ByteTrack](https://github.com/ifzhang/ByteTrack), 
[FastReID](https://github.com/JDAI-CV/fast-reid),
[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) and
[FastMOT](https://github.com/GeekAlexis/FastMOT).
Thanks for their excellent work!