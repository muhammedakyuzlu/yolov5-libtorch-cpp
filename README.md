## Introduction

A LibTorch inference implementation of the [yolov5](https://github.com/ultralytics/yolov5) object detection algorithm. only CPU is supported.



## Dependencies

- Ubuntu
- OpenCV 
- LibTorch




## Setup

```bash
$ cd /path/to/libtorch_yolo5
$ wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.10.2%2Bcpu.zip
$ unzip libtorch-cxx11-abi-shared-with-deps-1.10.2+cpu.zip
$ mkdir build && cd build
$ cmake .. && make
```





# run

```bash
   ./libtorch-yolov5 [OPTION...]

   --weights arg      path to model.torchscript.pt
   --classes arg      path to classes name  coco.name
   -i                 image or video/stream default:video
   --source arg       path to source
   --conf-thresh arg  object confidence threshold (default: 0.4)
   --iou-thresh arg   IOU threshold for NMS (default: 0.5)
   --show             display results
   -h, --help         Print usage
```


### video
```bash
./libtorch-yolov5 --weights ../weights/yolov5s.torchscript --classes ../weights/coco.names --source ../videos/t2.mp4 [--show] [--conf-thresh] [--iou-thresh]
```

### camera by index
```bash
./libtorch-yolov5 --weights ../weights/yolov5s.torchscript --classes ../weights/coco.names --source 0 [--show] [--conf-thresh] [--iou-thresh]
```


### image
```
./libtorch-yolov5 --weights ../weights/yolov5s.torchscript --classes ../weights/coco.names --source ../images/bus.jpg [--show] [--conf-thresh] [--iou-thresh]
```