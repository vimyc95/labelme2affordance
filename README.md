# Convert labelme to affordance (IIT-AFF dataset format)

## Installation
```bash
conda create -n py3.10 python==3.10
conda activate py3.10
pip install labelme
pip install opencv-python-headless==4.12.0.88
```

## Usage
### Example
```bash
python labelme2aff.py --input_dir data/  --output_dir data_aff/
```
## Annotation Method
The following image illustrates the annotation process:

<!-- ![](assert/example.png) -->
<img src="assert/example.png" width="800">

### label
```
${obj_number}_{object_name}_{affordance_part_name}

# example
1_hammer_grasp
1_hammer_pound
2_hammer_grasp
2_hammer_pound
3_knife_cut
3_knife_grasp
...
```

### Output
After running the script, the following will be generated:
* The original RGB image
* The object bounding box
* Individual affordance masks for each object
 

### Result Preview
Note: All masks shown here are multiplied by 30 to enhance pixel intensity for better visualization.
|RGB Image|Object 1|
:---:|:---:|
| ![](assert/img0001.jpg) | ![](assert/img0001_1_segmask.png) |
|Object 2|Object 3|
| ![](assert/img0001_2_segmask.png) | ![](assert/img0001_3_segmask.png) |