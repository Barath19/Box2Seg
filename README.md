# Box2Seg
Box2Seg is a semantic annotation tool but faster.


![image tool](assets/tool.png)


# Why another annotation tool

Deep learning-based computer vision models are becoming more prominent with various applications
in medical imaging, automotive, logistics, etc. However, they require a large amount of annotated
data to perform significantly well, and generating annotation labels for these large datasets is
a cumbersome and time-consuming task. Specifically, for segmentation applications generating
annotated data is a relatively difficult task compared to drawing simple bounding boxes for object
detection. For example, bounding box labels require just two points, whereas semantic labels
require a minimum of 20 points, even for a coarse label, as shown in Figure 1 (manually generated
using labelme annotation tool1)

![image tool](assets/output.png)

