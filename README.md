# PineSORT: A Simple Online Real-time Tracking Framework for Drone Videos in Agriculture

## Abstract
We introduce PineSORT, a novel Multiple Object Tracking (MOT) system for drone-based agricultural monitoring, specifically tracking pineapples for yield estimation. Our approach tackles key challenges such as repetitive patterns, similar object appearances, low frame rates, and drone motion effects. PineSORT enhances the tracking accuracy with motion direction cost, camera motion compensation, a three-stage association strategy, and overlap management. To handle large displacements, we propose an ORB-based camera compensation technique that significantly improves the Association Accuracy (AssA). Evaluated via 5-fold cross-validation against BoTSORT and AgriSORT, PineSORT achieves statistically significant gains in our Identity-Switch Penalized IDF1 (ISP-IDF1) metric, along with gains in IDF1 (Identity F1 Score), HOTA (Higher Order Tracking Accuracy) and AssA. These results confirm its effectiveness in tracking low-FPS drone footage, making it a valuable tool for precision agriculture.


## 📖 Citation

If you find this repository useful, please star ⭐ the repository and cite:

```
@InProceedings{Xie-Li_2025_CVPR,
    author    = {Xie-Li, Danny and Fallas-Moya, Fabian},
    title     = {PineSORT: A Simple Online Real-time Tracking Framework for Drone Videos in Agriculture},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR) Workshops},
    month     = {June},
    year      = {2025},
    pages     = {65-74}
}
```
