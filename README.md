<!-- # Awesome-single-object-tracking

<!-- # Awesome Object Pose Estimation and Reconstruction [![Awesome](https://awesome.re/badge-flat.svg)](https://awesome.re) !-->
<!-- [![Maintenance](https://img.shields.io/badge/Maintained%3F-YES-green.svg)](https://github.com/memoryunreal/awesome-single-object-tracking/graphs/commit-activity)
![ ](https://img.shields.io/github/last-commit/memoryunreal/awesome-single-object-tracking)
[![GitHub stars](https://img.shields.io/github/stars/MinghuiChen43/awesome-trustworthy-deep-learning?color=blue&style=plastic)](https://github.com/MinghuiChen43/awesome-trustworthy-deep-learning/stargazers)
[![GitHub watchers](https://img.shields.io/github/watchers/MinghuiChen43/awesome-trustworthy-deep-learning?color=yellow&style=plastic)](https://github.com/MinghuiChen43/awesome-trustworthy-deep-learning)
[![GitHub forks](https://img.shields.io/github/forks/MinghuiChen43/awesome-trustworthy-deep-learning?color=red&style=plastic)](https://github.com/MinghuiChen43/awesome-trustworthy-deep-learning/watchers)
[![GitHub Contributors](https://img.shields.io/github/contributors/MinghuiChen43/awesome-trustworthy-deep-learning?color=green&style=plastic)](https://github.com/MinghuiChen43/awesome-trustworthy-deep-learning/network/members) -->
<a id="markdown-contents" name="contents"></a>

# Contents
Related papers will be continuously updated. Please feel free to contact [liz8@mail.sustech.edu.cn](mailto:liz8@mail.sustech.edu.cn), [jinyu.yang96@outlook.com](mailto:jinyu.yang96@outlook.com), and [gaos2021@mail.sustech.edu.cn](mailto:gaos2021@mail.sustech.edu.cn).

BTW: Welcome to visit the SUSTech-VIP Lab as research assistants and visiting students. Looking forward to working with you! [[VIPG-Lab]](https://sustech-vip-lab.github.io/Visual%20Anomaly%20Detection/) [[加入我们]](https://faculty.sustech.edu.cn/?cat=11&tagid=fengzheng&orderby=date&iscss=1&snapid=1) [[Join us]](https://faculty.sustech.edu.cn/?cat=11&tagid=fengzheng&orderby=date&iscss=1&snapid=1&lang=en)
<!-- TOC -->
  - [Review papers]
    * **Survey**: Know Your Surroundings: Exploiting Scene Information for Object Tracking. In _arxiv_ 2020. [[Paper]](https://arxiv.org/pdf/2003.11014v1.pdf)
    ### **Transformer**
    - [**2022**](#2022)
      - Joint Feature Learning and Relation Modeling for Tracking: A One-Stream Framework. [[paper]](https://arxiv.org/abs/2203.11991) [[code]](https://github.com/botaoye/ostrack)
        - Botao Ye, Hong Chang, Bingpeng Ma, Shiguang Shan. *ECCV 2022*
        - Keyword: One-stream; Target-background Discriminability;
        - <details><summary>Digest</summary> The current popular two-stream, two-stage tracking framework extracts the template and the search region features separately and then performs relation modeling, thus the extracted features lack the awareness of the target and have limited target-background discriminability. To tackle the above issue, we propose a novel one-stream tracking (OSTrack) framework that unifies feature learning and relation modeling by bridging the template-search image pairs with bidirectional information flows. In this way, discriminative target-oriented features can be dynamically extracted by mutual guidance. Since no extra heavy relation modeling module is needed and the implementation is highly parallelized, the proposed tracker runs at a fast speed. To further improve the inference efficiency, an in-network candidate early elimination module is proposed based on the strong similarity prior calculated in the one-stream framework. As a unified framework, OSTrack achieves state-of-the-art performance on multiple benchmarks, in particular, it shows impressive results on the one-shot tracking benchmark GOT-10k, i.e., achieving 73.7% AO, improving the existing best result (SwinTrack) by 4.3%. Besides, our method maintains a good performance-speed trade-off and shows faster convergence.
        - ![A](https://github.com/memoryunreal/awesome-single-object-tracking/blob/main/Figure/OSTrack.png)
        - <img src="https://github.com/memoryunreal/awesome-single-object-tracking/blob/main/Figure/OSTrack.png" /> 

      - MixFormer: End-to-End Tracking with Iterative Mixed Attention. [[paper]](https://arxiv.org/abs/2203.11082) [[code]](https://github.com/MCG-NJU/MixFormer)
        - Yutao Cui, Cheng Jiang, Limin Wang, Gangshan Wu. *CVPR 2022 Oral*
        - Keyword: Transformer; Mixed Attention Module.
        - <details><summary>Digest</summary> To simplify the tracking pipeline and unify the process of feature extraction and target information integration, we present a compact tracking framework, termed as MixFormer, built upon transformers. Our core design is to utilize the flexibility of attention operations, and propose a Mixed Attention Module (MAM) for simultaneous feature extraction and target information integration. This synchronous modeling scheme allows to extract target-specific discriminative features and perform extensive communication between target and search area. Based on MAM, we build our MixFormer tracking framework simply by stacking multiple MAMs with progressive patch embedding and placing a localization head on top. In addition, to handle multiple target templates during online tracking, we devise an asymmetric attention scheme in MAM to reduce computational cost, and propose an effective score prediction module to select high-quality templates. Our MixFormer sets a new state-of-the-art performance on five tracking benchmarks, including LaSOT, TrackingNet, VOT2020, GOT-10k, and UAV123. In particular, our MixFormer-L achieves NP score of 79.9% on LaSOT, 88.9% on TrackingNet and EAO of 0.555 on VOT2020. We also perform in-depth ablation studies to demonstrate the effectiveness of simultaneous feature extraction and information integration.

    - [**2021**](#2021)

      * **STARK**: Yan, Bin and Peng, Houwen and Fu, Jianlong and Wang, Dong and Lu, Huchuan. Learning Spatio-Temporal Transformer for Visual Tracking. In _ICCV_ 2021.[[Paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Yan_Learning_Spatio-Temporal_Transformer_for_Visual_Tracking_ICCV_2021_paper.pdf) [[Code]](https://github.com/researchmm/Stark)
      * **Transformer Tracking**: Chen, Xin and Yan, Bin and Zhu, Jiawen and Wang, Dong and Yang, Xiaoyun and Lu, Huchuan. Transformer Tracking. In _CVPR_ 2021.[[Paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Transformer_Tracking_CVPR_2021_paper.pdf) [[Code]](https://github.com/chenxin-dlut/TransT)
      * **Transformer Meets Tracker**: Wang, Ning and Zhou, Wengang and Wang, Jie and Li, Houqiang. Transformer Meets Tracker: Exploiting Temporal Context for Robust Visual Tracking. In _CVPR_ 2021. [[Paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_Transformer_Meets_Tracker_Exploiting_Temporal_Context_for_Robust_Visual_Tracking_CVPR_2021_paper.pdf) [[Code]](https://github.com/594422814/TransformerTrack)
      * High-Performance Discriminative Tracking With Transformers. In _ICCV_ 2021. [[Paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Yu_High-Performance_Discriminative_Tracking_With_Transformers_ICCV_2021_paper.pdf)
      * **HiFT**: Cao, Ziang and Fu, Changhong and Ye, Junjie and Li, Bowen and Li, Yiming. HiFT: Hierarchical Feature Transformer for Aerial Tracking. In _ICCV_ 2021. [[Paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Cao_HiFT_Hierarchical_Feature_Transformer_for_Aerial_Tracking_ICCV_2021_paper.pdf) [[Code]](https://github.com/vision4robotics/HiFT)
     


    ### **Siamese**
    - [**2021**](#2021)
      * **STMTrack**: Fu, Zhihong and Liu, Qingjie and Fu, Zehua and Wang, Yunhong. STMTrack: Template-Free Visual Tracking With Space-Time Memory Networks. In _CVPR_ 2021. [[Paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Fu_STMTrack_Template-Free_Visual_Tracking_With_Space-Time_Memory_Networks_CVPR_2021_paper.pdf) [[Code]](https://github.com/fzh0917/STMTrack)
      * **LightTrack**: Yan, Bin and Peng, Houwen and Wu, Kan and Wang, Dong and Fu, Jianlong and Lu, Huchuan. LightTrack: Finding Lightweight Neural Networks for Object Tracking via One-Shot Architecture Search. In _CVPR_ 2021. [[Paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Yan_LightTrack_Finding_Lightweight_Neural_Networks_for_Object_Tracking_via_One-Shot_CVPR_2021_paper.pdf) [[Code]](https://github.com/researchmm/LightTrack)
      * **Alpha-Refine**: Yan, Bin and Zhang, Xinyu and Wang, Dong and Lu, Huchuan and Yang, Xiaoyun. Alpha-Refine: Boosting Tracking Performance by Precise Bounding Box Estimation. In _CVPR_ 2021. [[Paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Yan_Alpha-Refine_Boosting_Tracking_Performance_by_Precise_Bounding_Box_Estimation_CVPR_2021_paper.pdf) [[Code]](https://github.com/MasterBin-IIAU/AlphaRefine)
    - [**2020**](#2020)
      * **Siam R-CNN:** Paul Voigtlaender, Jonathon Luiten, Philip H.S. Torr, Bastian Leibe. Siam R-CNN: Visual Tracking by Re-Detection. In _CVPR_ 2020.
  [Paper](https://arxiv.org/pdf/1911.12836.pdf) [[Code]](https://www.vision.rwth-aachen.de/page/siamrcnn)
    - [**2019**](#2019)
      * **SiamRpn++**: Li, Bo, Wei Wu, Qiang Wang, Fangyi Zhang, Junliang Xing, and Junjie Yan. Siamrpn++: Evolution of siamese visual tracking with very deep networks. In _CVPR_ 2019. [[Paper]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_SiamRPN_Evolution_of_Siamese_Visual_Tracking_With_Very_Deep_Networks_CVPR_2019_paper.pdf) [[Project]](https://lb1100.github.io/SiamRPN++/)
    - [**2017**](#2017)
      * **CFNet**: Valmadre, Jack, Luca Bertinetto, Joao Henriques, Andrea Vedaldi, and Philip HS Torr. End-to-end representation learning for correlation filter based tracking. In _CVPR_ 2017. [[Paper]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Valmadre_End-To-End_Representation_Learning_CVPR_2017_paper.pdf) [[Project]](http://www.robots.ox.ac.uk/~luca/cfnet.html)
    - [**2016**](#2016)
      * **SiamFC**: Bertinetto, Luca, Jack Valmadre, Joao F. Henriques, Andrea Vedaldi, and Philip HS Torr. Fully-convolutional siamese networks for object tracking. In _ECCV_ 2016. [[Paper]](https://arxiv.org/pdf/1606.09549.pdf) [[Project]](https://www.robots.ox.ac.uk/~luca/siamese-fc.html)
      
    ### **Point Tracking**
    - [**2021**](#2021)
      * Zheng, Chaoda and Yan, Xu and Gao, Jiantao and Zhao, Weibing and Zhang, Wei and Li, Zhen and Cui, Shuguang. Box-Aware Feature Enhancement for Single Object Tracking on Point Clouds. In _ICCV_ 2021. [[Paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Zheng_Box-Aware_Feature_Enhancement_for_Single_Object_Tracking_on_Point_Clouds_ICCV_2021_paper.pdf) [[Code]](https://github.com/Ghostish/BAT)
    - [**2020**](#2020)
      * Qi, Haozhe, Chen Feng, Zhiguo Cao, Feng Zhao, and Yang Xiao. P2B: Point-to-box network for 3D object tracking in point clouds. In _CVPR_ 2020. [[Paper]](https://openaccess.thecvf.com/content_CVPR_2020/papers/Qi_P2B_Point-to-Box_Network_for_3D_Object_Tracking_in_Point_Clouds_CVPR_2020_paper.pdf) [[Code]](https://github.com/HaozheQi/P2B)
    - [**2019**](#2019)
      * Giancola, Silvio, Jesus Zarzar, and Bernard Ghanem. Leveraging shape completion for 3d siamese tracking. In _CVPR_ 2019. [[Paper]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Giancola_Leveraging_Shape_Completion_for_3D_Siamese_Tracking_CVPR_2019_paper.pdf) [[Code]](https://github.com/SilvioGiancola/ShapeCompletion3DTracking)
    ## **RGB Paper**
    ### CVPR2020

* **MAML:** Guangting Wang, Chong Luo, Xiaoyan Sun, Zhiwei Xiong, Wenjun Zeng.<br />
  "Tracking by Instance Detection: A Meta-Learning Approach." CVPR (2020 **Oral**).
  [[paper](https://arxiv.org/pdf/2004.00830v1.pdf)]

* **Siam R-CNN:** Paul Voigtlaender, Jonathon Luiten, Philip H.S. Torr, Bastian Leibe.<br />
  "Siam R-CNN: Visual Tracking by Re-Detection." CVPR (2020).
  [[paper](https://arxiv.org/pdf/1911.12836.pdf)] 
  [[code](https://www.vision.rwth-aachen.de/page/siamrcnn)]

* **D3S:** Alan Lukežič, Jiří Matas, Matej Kristan.<br />
  "D3S – A Discriminative Single Shot Segmentation Tracker." CVPR (2020).
  [[paper](http://arxiv.org/pdf/1911.08862v2.pdf)]
  [[code](https://github.com/alanlukezic/d3s)]

* **PrDiMP:** Martin Danelljan, Luc Van Gool, Radu Timofte.<br />
  "Probabilistic Regression for Visual Tracking." CVPR (2020).
  [[paper](https://arxiv.org/pdf/2003.12565v1.pdf)]
  [[code](https://github.com/visionml/pytracking)]

* **ROAM:** Tianyu Yang, Pengfei Xu, Runbo Hu, Hua Chai, Antoni B. Chan.<br />
  "ROAM: Recurrently Optimizing Tracking Model." CVPR (2020).
  [[paper](https://arxiv.org/pdf/1907.12006v3.pdf)]

* **AutoTrack:** Yiming Li, Changhong Fu, Fangqiang Ding, Ziyuan Huang, Geng Lu.<br />
  "AutoTrack: Towards High-Performance Visual Tracking for UAV with Automatic Spatio-Temporal Regularization." CVPR (2020).
  [[paper](https://arxiv.org/pdf/2003.12949.pdf)]
  [[code](https://github.com/vision4robotics/AutoTrack)]

* **SiamBAN:** Zedu Chen, Bineng Zhong, Guorong Li, Shengping Zhang, Rongrong Ji.<br />
  "Siamese Box Adaptive Network for Visual Tracking." CVPR (2020).
  [[paper](http://arxiv.org/pdf/1911.08862v2.pdf)]
  [[code](https://github.com/hqucv/siamban)]

* **SiamAttn:** Yuechen Yu, Yilei Xiong, Weilin Huang, Matthew R. Scott. <br />
  "Deformable Siamese Attention Networks for Visual Object Tracking." CVPR (2020).
  [[paper](https://arxiv.org/pdf/2004.06711v1.pdf)]

* **CGACD:** Fei Du, Peng Liu, Wei Zhao, Xianglong Tang.<br />
  "Correlation-Guided Attention for Corner Detection Based Visual Tracking." CVPR (2020).


### AAAI 2020

- **SiamFC++:** Yinda Xu, Zeyu Wang, Zuoxin Li, Ye Yuan, Gang Yu. <br />
  "SiamFC++: Towards Robust and Accurate Visual Tracking with Target Estimation Guidelines." AAAI (2020).
  [[paper](https://arxiv.org/pdf/1911.06188v4.pdf)]
  [[code](https://github.com/MegviiDetection/video_analyst)]


### ICCV2019

* **DiMP:** Goutam Bhat, Martin Danelljan, Luc Van Gool, Radu Timofte.<br />
  "Learning Discriminative Model Prediction for Tracking." ICCV (2019 **oral**). 
  [[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Bhat_Learning_Discriminative_Model_Prediction_for_Tracking_ICCV_2019_paper.pdf)]
  [[code](https://github.com/visionml/pytracking)]

* **GradNet:** Peixia Li, Boyu Chen, Wanli Ouyang, Dong Wang, Xiaoyun Yang, Huchuan Lu. <br />
  "GradNet: Gradient-Guided Network for Visual Object Tracking." ICCV (2019 **oral**).
  [[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Li_GradNet_Gradient-Guided_Network_for_Visual_Object_Tracking_ICCV_2019_paper.pdf)]
  [[code](https://github.com/LPXTT/GradNet-Tensorflow)]

* **MLT:** Janghoon Choi, Junseok Kwon, Kyoung Mu Lee. <br />
  "Deep Meta Learning for Real-Time Target-Aware Visual Tracking." ICCV (2019).
  [[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Choi_Deep_Meta_Learning_for_Real-Time_Target-Aware_Visual_Tracking_ICCV_2019_paper.pdf)]

* **SPLT:** Bin Yan, Haojie Zhao, Dong Wang, Huchuan Lu, Xiaoyun Yang <br />
  "'Skimming-Perusal' Tracking: A Framework for Real-Time and Robust Long-Term Tracking." ICCV (2019).
  [[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Yan_Skimming-Perusal_Tracking_A_Framework_for_Real-Time_and_Robust_Long-Term_Tracking_ICCV_2019_paper.pdf)]
  [[code](https://github.com/iiau-tracker/SPLT)]

* **ARCF:** Ziyuan Huang, Changhong Fu, Yiming Li, Fuling Lin, Peng Lu. <br />
  "Learning Aberrance Repressed Correlation Filters for Real-Time UAV Tracking." ICCV (2019).
  [[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Huang_Learning_Aberrance_Repressed_Correlation_Filters_for_Real-Time_UAV_Tracking_ICCV_2019_paper.pdf)]
  [[code](https://github.com/vision4robotics/ARCF-tracker)]

* Lianghua Huang, Xin Zhao, Kaiqi Huang. <br />
  "Bridging the Gap Between Detection and Tracking: A Unified Approach." ICCV (2019).
  [[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Huang_Bridging_the_Gap_Between_Detection_and_Tracking_A_Unified_Approach_ICCV_2019_paper.pdf)]

* **UpdateNet:** Lichao Zhang, Abel Gonzalez-Garcia, Joost van de Weijer, Martin Danelljan, Fahad Shahbaz Khan. <br />
  "Learning the Model Update for Siamese Trackers." ICCV (2019).
  [[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Zhang_Learning_the_Model_Update_for_Siamese_Trackers_ICCV_2019_paper.pdf)]
  [[code](https://github.com/zhanglichao/updatenet)]

* **PAT:** Rey Reza Wiyatno, Anqi Xu. <br />
  "Physical Adversarial Textures That Fool Visual Object Tracking." ICCV (2019).
  [[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Wiyatno_Physical_Adversarial_Textures_That_Fool_Visual_Object_Tracking_ICCV_2019_paper.pdf)]

* **GFS-DCF:** Tianyang Xu, Zhen-Hua Feng, Xiao-Jun Wu, Josef Kittler. <br />
  "Joint Group Feature Selection and Discriminative Filter Learning for Robust Visual Object Tracking." ICCV (2019).
  [[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Xu_Joint_Group_Feature_Selection_and_Discriminative_Filter_Learning_for_Robust_ICCV_2019_paper.pdf)]
  [[code](https://github.com/XU-TIANYANG/GFS-DCF)]

* **CDTB:** Alan Lukežič, Ugur Kart, Jani Käpylä, Ahmed Durmush, Joni-Kristian Kämäräinen, Jiří Matas, Matej Kristan. <br />

  "CDTB: A Color and Depth Visual Object Tracking Dataset and Benchmark." ICCV (2019).
  [[paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Lukezic_CDTB_A_Color_and_Depth_Visual_Object_Tracking_Dataset_and_ICCV_2019_paper.pdf)]

* **VOT2019:** Kristan, Matej, et al.<br />
  "The Seventh Visual Object Tracking VOT2019 Challenge Results." ICCV workshops (2019).
  [[paper](http://openaccess.thecvf.com/content_ICCVW_2019/papers/VOT/Kristan_The_Seventh_Visual_Object_Tracking_VOT2019_Challenge_Results_ICCVW_2019_paper.pdf)]


### CVPR2019

* **SiamMask:** Qiang Wang, Li Zhang, Luca Bertinetto, Weiming Hu, Philip H.S. Torr.<br />
  "Fast Online Object Tracking and Segmentation: A Unifying Approach." CVPR (2019). 
  [[paper](https://arxiv.org/pdf/1812.05050.pdf)]
  [[project](http://www.robots.ox.ac.uk/~qwang/SiamMask/)]
  [[code](https://github.com/foolwood/SiamMask)]

* **SiamRPN++:** Bo Li, Wei Wu, Qiang Wang, Fangyi Zhang, Junliang Xing, Junjie Yan.<br />
  "SiamRPN++: Evolution of Siamese Visual Tracking with Very Deep Networks." CVPR (2019 **oral**). 
  [[paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Li_SiamRPN_Evolution_of_Siamese_Visual_Tracking_With_Very_Deep_Networks_CVPR_2019_paper.pdf)]
  [[project](http://bo-li.info/SiamRPN++/)]

* **ATOM:** Martin Danelljan, Goutam Bhat, Fahad Shahbaz Khan, Michael Felsberg. <br />
  "ATOM: Accurate Tracking by Overlap Maximization." CVPR (2019 **oral**). 
  [[paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Danelljan_ATOM_Accurate_Tracking_by_Overlap_Maximization_CVPR_2019_paper.pdf)]
  [[code](https://github.com/visionml/pytracking)]

* **SiamDW:** Zhipeng Zhang, Houwen Peng.<br />
  "Deeper and Wider Siamese Networks for Real-Time Visual Tracking." CVPR (2019 **oral**). 
  [[paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_Deeper_and_Wider_Siamese_Networks_for_Real-Time_Visual_Tracking_CVPR_2019_paper.pdf)]
  [[code](https://github.com/researchmm/SiamDW)]

* **GCT:** Junyu Gao, Tianzhu Zhang, Changsheng Xu.<br />
  "Graph Convolutional Tracking." CVPR (2019 **oral**).
  [[paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Gao_Graph_Convolutional_Tracking_CVPR_2019_paper.pdf)]
  [[code](https://github.com/researchmm/SiamDW)]

* **ASRCF:** Kenan Dai, Dong Wang, Huchuan Lu, Chong Sun, Jianhua Li. <br />
  "Visual Tracking via Adaptive Spatially-Regularized Correlation Filters." CVPR (2019 **oral**).
  [[paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Dai_Visual_Tracking_via_Adaptive_Spatially-Regularized_Correlation_Filters_CVPR_2019_paper.pdf)]
  [[code](https://github.com/Daikenan/ASRCF)]

* **UDT:** Ning Wang, Yibing Song, Chao Ma, Wengang Zhou, Wei Liu, Houqiang Li.<br />
  "Unsupervised Deep Tracking." CVPR (2019). 
  [[paper](https://arxiv.org/pdf/1904.01828.pdf)]
  [[code](https://github.com/594422814/UDT)]

* **TADT:** Xin Li, Chao Ma, Baoyuan Wu, Zhenyu He, Ming-Hsuan Yang.<br />
  "Target-Aware Deep Tracking." CVPR (2019). 
  [[paper](https://arxiv.org/pdf/1904.01772.pdf)]
  [[project](https://xinli-zn.github.io/TADT-project-page/)]
  [[code](https://github.com/XinLi-zn/TADT)]

* **C-RPN:** Heng Fan, Haibin Ling.<br />
  "Siamese Cascaded Region Proposal Networks for Real-Time Visual Tracking." CVPR (2019). 
  [[paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Fan_Siamese_Cascaded_Region_Proposal_Networks_for_Real-Time_Visual_Tracking_CVPR_2019_paper.pdf)]

* **SPM:** Guangting Wang, Chong Luo, Zhiwei Xiong, Wenjun Zeng.<br />
  "SPM-Tracker: Series-Parallel Matching for Real-Time Visual Object Tracking." CVPR (2019). 
  [[paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_SPM-Tracker_Series-Parallel_Matching_for_Real-Time_Visual_Object_Tracking_CVPR_2019_paper.pdf)]

* **OTR:** Ugur Kart, Alan Lukezic, Matej Kristan, Joni-Kristian Kamarainen, Jiri Matas. <br />
  "Object Tracking by Reconstruction with View-Specific Discriminative Correlation Filters." CVPR (2019). 
  [[paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Kart_Object_Tracking_by_Reconstruction_With_View-Specific_Discriminative_Correlation_Filters_CVPR_2019_paper.pdf)]
  [[code](https://github.com/ugurkart/OTR)]

* **RPCF:** Yuxuan Sun, Chong Sun, Dong Wang, Huchuan Lu, You He. <br />
  "ROI Pooled Correlation Filters for Visual Tracking." CVPR (2019).
  [[paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Sun_ROI_Pooled_Correlation_Filters_for_Visual_Tracking_CVPR_2019_paper.pdf)]

* **LaSOT:** Heng Fan, Liting Lin, Fan Yang, Peng Chu, Ge Deng, Sijia Yu, Hexin Bai, Yong Xu, Chunyuan Liao, Haibin Ling.<br />
  "LaSOT: A High-quality Benchmark for Large-scale Single Object Tracking." CVPR (2019). 
  [[paper](https://arxiv.org/pdf/1809.07845.pdf)]
  [[project](https://cis.temple.edu/lasot/)]

### AAAI2019

* **LDES:** Yang Li, Jianke Zhu, Steven C.H. Hoi, Wenjie Song, Zhefeng Wang, Hantang Liu.<br />
  "Robust Estimation of Similarity Transformation for Visual Object Tracking." AAAI (2019). 
  [[paper](https://arxiv.org/pdf/1712.05231.pdf)]
  [[code](https://github.com/ihpdep/LDES)] 

### NIPS2018

* **DAT:** Shi Pu, Yibing Song, Chao Ma, Honggang Zhang, Ming-Hsuan Yang.<br />
  "Deep Attentive Tracking via Reciprocative Learning." NIPS (2018). 
  [[paper](https://arxiv.org/pdf/1810.03851.pdf)] 
  [[project](https://ybsong00.github.io/nips18_tracking/index)] 
  [[code](https://github.com/shipubupt/NIPS2018)] 

### ECCV2018

* **UPDT:** Goutam Bhat, Joakim Johnander, Martin Danelljan, Fahad Shahbaz Khan, Michael Felsberg.<br />
  "Unveiling the Power of Deep Tracking." ECCV (2018). 
  [[paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Goutam_Bhat_Unveiling_the_Power_ECCV_2018_paper.pdf)]  

* **DaSiamRPN:** Zheng Zhu, Qiang Wang, Bo Li, Wu Wei, Junjie Yan, Weiming Hu.<br />
  "Distractor-aware Siamese Networks for Visual Object Tracking." ECCV (2018).
  [[paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Zheng_Zhu_Distractor-aware_Siamese_Networks_ECCV_2018_paper.pdf)]
  [[github](https://github.com/foolwood/DaSiamRPN)]

* **SACF:** Mengdan Zhang, Qiang Wang, Junliang Xing, Jin Gao, Peixi Peng, Weiming Hu, Steve Maybank.<br />
  "Visual Tracking via Spatially Aligned Correlation Filters Network." ECCV (2018).
  [[paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/mengdan_zhang_Visual_Tracking_via_ECCV_2018_paper.pdf)]

* **RTINet:** Yingjie Yao, Xiaohe Wu, Lei Zhang, Shiguang Shan, Wangmeng Zuo.<br />
  "Joint Representation and Truncated Inference Learning for Correlation Filter based Tracking." ECCV (2018).
  [[paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Yingjie_Yao_Joint_Representation_and_ECCV_2018_paper.pdf)]

* **Meta-Tracker:** Eunbyung Park, Alexander C. Berg.<br />
  "Meta-Tracker: Fast and Robust Online Adaptation for Visual Object Trackers."
  [[paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Eunbyung_Park_Meta-Tracker_Fast_and_ECCV_2018_paper.pdf)]
  [[github](https://github.com/silverbottlep/meta_trackers)]

* **DSLT:** Xiankai Lu, Chao Ma*, Bingbing Ni, Xiaokang Yang, Ian Reid, Ming-Hsuan Yang.<br />
  "Deep Regression Tracking with Shrinkage Loss." ECCV (2018).
  [[paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Xiankai_Lu_Deep_Regression_Tracking_ECCV_2018_paper.pdf)]
  [[github](https://github.com/chaoma99/DSLT)]

* **DRL-IS:** Liangliang Ren, Xin Yuan, Jiwen Lu, Ming Yang, Jie Zhou.<br />
  "Deep Reinforcement Learning with Iterative Shift for Visual Tracking." ECCV (2018).
  [[paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Liangliang_Ren_Deep_Reinforcement_Learning_ECCV_2018_paper.pdf)]

* **RT-MDNet:** Ilchae Jung, Jeany Son, Mooyeol Baek, Bohyung Han.<br />
  "Real-Time MDNet." ECCV (2018).
  [[paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Ilchae_Jung_Real-Time_MDNet_ECCV_2018_paper.pdf)]

* **ACT:** Boyu Chen, Dong Wang, Peixia Li, Huchuan Lu.<br />
  "Real-time 'Actor-Critic' Tracking." ECCV (2018).
  [[paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Boyu_Chen_Real-time_Actor-Critic_Tracking_ECCV_2018_paper.pdf)]
  [[github](https://github.com/bychen515/ACT)]

* **StructSiam:** Yunhua Zhang, Lijun Wang, Dong Wang, Mengyang Feng, Huchuan Lu, Jinqing Qi.<br />
  "Structured Siamese Network for Real-Time Visual Tracking." ECCV (2018).
  [[paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Yunhua_Zhang_Structured_Siamese_Network_ECCV_2018_paper.pdf)]

* **MemTrack:** Tianyu Yang, Antoni B. Chan.<br />
  "Learning Dynamic Memory Networks for Object Tracking." ECCV (2018).
  [[paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Tianyu_Yang_Learning_Dynamic_Memory_ECCV_2018_paper.pdf)]

* **SiamFC-tri:** Xingping Dong, Jianbing Shen.<br />
  "Triplet Loss in Siamese Network for Object Tracking." ECCV (2018).
  [[paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Xingping_Dong_Triplet_Loss_with_ECCV_2018_paper.pdf)]
  [[github](https://github.com/shenjianbing/TripletTracking)]

* **OxUvA long-term dataset+benchmark:** Jack Valmadre, Luca Bertinetto, João F. Henriques, Ran Tao, Andrea Vedaldi, Arnold Smeulders, Philip Torr, Efstratios Gavves.<br />
  "Long-term Tracking in the Wild: a Benchmark." ECCV (2018).
  [[paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Efstratios_Gavves_Long-term_Tracking_in_ECCV_2018_paper.pdf)]
  [[project](https://oxuva.github.io/long-term-tracking-benchmark/)]

* **TrackingNet:** Matthias Müller, Adel Bibi, Silvio Giancola, Salman Al-Subaihi, Bernard Ghanem.<br />
  "TrackingNet: A Large-Scale Dataset and Benchmark for Object Tracking in the Wild." ECCV (2018).
  [[paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Matthias_Muller_TrackingNet_A_Large-Scale_ECCV_2018_paper.pdf)] 
  [[project](http://tracking-net.org/)]


### CVPR2018

* **VITAL:** Yibing Song, Chao Ma, Xiaohe Wu, Lijun Gong, Linchao Bao, Wangmeng Zuo, Chunhua Shen, Rynson Lau, and Ming-Hsuan Yang.
  "VITAL: VIsual Tracking via Adversarial Learning." CVPR (2018 **Spotlight**). 
  [[project](https://ybsong00.github.io/cvpr18_tracking/index)]
  [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Song_VITAL_VIsual_Tracking_CVPR_2018_paper.pdf)]
  [[github](https://github.com/ybsong00/Vital_release)]

* **LSART:** Chong Sun, Dong Wang, Huchuan Lu, Ming-Hsuan Yang.
  "Learning Spatial-Aware Regressions for Visual Tracking." CVPR (2018 **Spotlight**). 
  [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Sun_Learning_Spatial-Aware_Regressions_CVPR_2018_paper.pdf)]

* **SiamRPN:** Bo Li, Wei Wu, Zheng Zhu, Junjie Yan.
  "High Performance Visual Tracking with Siamese Region Proposal Network." CVPR (2018 **Spotlight**). 
  [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Li_High_Performance_Visual_CVPR_2018_paper.pdf)]

* **TRACA:** Jongwon Choi, Hyung Jin Chang, Tobias Fischer, Sangdoo Yun, Kyuewang Lee, Jiyeoup Jeong, Yiannis Demiris, Jin Young Choi.
  "Context-aware Deep Feature Compression for High-speed Visual Tracking." CVPR (2018). 
  [[project](https://sites.google.com/site/jwchoivision/)]
  [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Choi_Context-Aware_Deep_Feature_CVPR_2018_paper.pdf)]

* **RASNet:** Qiang Wang, Zhu Teng, Junliang Xing, Jin Gao, Weiming Hu, Stephen Maybank.
  "Learning Attentions: Residual Attentional Siamese Network for High Performance Online Visual Tracking." CVPR 2018. 
  [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Learning_Attentions_Residual_CVPR_2018_paper.pdf)]

* **SA-Siam:** Anfeng He, Chong Luo, Xinmei Tian, Wenjun Zeng.
  "A Twofold Siamese Network for Real-Time Object Tracking." CVPR (2018). 
  [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/He_A_Twofold_Siamese_CVPR_2018_paper.pdf)]

* **STRCF:** Feng Li, Cheng Tian, Wangmeng Zuo, Lei Zhang, Ming-Hsuan Yang.
  "Learning Spatial-Temporal Regularized Correlation Filters for Visual Tracking." CVPR (2018). 
  [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Li_Learning_Spatial-Temporal_Regularized_CVPR_2018_paper.pdf)]
  [[github](https://github.com/lifeng9472/STRCF)]

* **FlowTrack:** Zheng Zhu, Wei Wu, Wei Zou, Junjie Yan.
  "End-to-end Flow Correlation Tracking with Spatial-temporal Attention." CVPR (2018). 
  [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhu_End-to-End_Flow_Correlation_CVPR_2018_paper.pdf)]

* **DEDT:** Kourosh Meshgi, Shigeyuki Oba, Shin Ishii.
  "Efficient Diverse Ensemble for Discriminative Co-Tracking." CVPR (2018). 
  [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Meshgi_Efficient_Diverse_Ensemble_CVPR_2018_paper.pdf)]

* **SINT++:** Xiao Wang, Chenglong Li, Bin Luo, Jin Tang.
  "SINT++: Robust Visual Tracking via Adversarial Positive Instance Generation." CVPR (2018).
  [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_SINT_Robust_Visual_CVPR_2018_paper.pdf)]

* **DRT:** Chong Sun, Dong Wang, Huchuan Lu, Ming-Hsuan Yang.
  "Correlation Tracking via Joint Discrimination and Reliability Learning." CVPR (2018). 
  [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Sun_Correlation_Tracking_via_CVPR_2018_paper.pdf)]

* **MCCT:** Ning Wang, Wengang Zhou, Qi Tian, Richang Hong, Meng Wang, Houqiang Li.
  "Multi-Cue Correlation Filters for Robust Visual Tracking." CVPR (2018). 
  [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Multi-Cue_Correlation_Filters_CVPR_2018_paper.pdf)]
  [[github](https://github.com/594422814/MCCT)]

* **MKCF:** Ming Tang, Bin Yu, Fan Zhang, Jinqiao Wang.
  "High-speed Tracking with Multi-kernel Correlation Filters." CVPR (2018).
  [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Tang_High-Speed_Tracking_With_CVPR_2018_paper.pdf)]

* **HP:** Xingping Dong, Jianbing Shen, Wenguan Wang, Yu, Liu, Ling Shao, and Fatih Porikli.
  "Hyperparameter Optimization for Tracking with Continuous Deep Q-Learning." CVPR (2018).
  [[paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Dong_Hyperparameter_Optimization_for_CVPR_2018_paper.pdf)]


### NIPS2017

* **HART:** Adam R. Kosiorek, Alex Bewley, Ingmar Posner. 
  "Hierarchical Attentive Recurrent Tracking." NIPS (2017). 
  [[paper](https://papers.nips.cc/paper/6898-hierarchical-attentive-recurrent-tracking.pdf)]
  [[github](https://github.com/akosiorek/hart)]


### ICCV2017

* **CREST:** Yibing Song, Chao Ma, Lijun Gong, Jiawei Zhang, Rynson Lau, Ming-Hsuan Yang. 
  "CREST: Convolutional Residual Learning for Visual Tracking." ICCV (2017 **Spotlight**). 
  [[paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Song_CREST_Convolutional_Residual_ICCV_2017_paper.pdf)]
  [[project](http://www.cs.cityu.edu.hk/~yibisong/iccv17/index.html)]
  [[github](https://github.com/ybsong00/CREST-Release)]

* **EAST:** Chen Huang, Simon Lucey, Deva Ramanan.
  "Learning Policies for Adaptive Tracking with Deep Feature Cascades." ICCV (2017 **Spotlight**). 
  [[paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Huang_Learning_Policies_for_ICCV_2017_paper.pdf)]
  [[supp](http://openaccess.thecvf.com/content_ICCV_2017/supplemental/Huang_Learning_Policies_for_ICCV_2017_supplemental.zip)]

* **PTAV:** Heng Fan and Haibin Ling. 
  "Parallel Tracking and Verifying: A Framework for Real-Time and High Accuracy Visual Tracking." ICCV (2017). 
  [[paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Fan_Parallel_Tracking_and_ICCV_2017_paper.pdf)]
  [[supp](http://openaccess.thecvf.com/content_ICCV_2017/supplemental/Fan_Parallel_Tracking_and_ICCV_2017_supplemental.pdf)]
  [[project](http://www.dabi.temple.edu/~hbling/code/PTAV/ptav.htm)]
  [[code](http://www.dabi.temple.edu/~hbling/code/PTAV/serial_ptav_v1.zip)]

* **BACF:** Hamed Kiani Galoogahi, Ashton Fagg, Simon Lucey. 
  "Learning Background-Aware Correlation Filters for Visual Tracking." ICCV (2017). 
  [[paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Galoogahi_Learning_Background-Aware_Correlation_ICCV_2017_paper.pdf)]
  [[supp](http://openaccess.thecvf.com/content_ICCV_2017/supplemental/Galoogahi_Learning_Background-Aware_Correlation_ICCV_2017_supplemental.pdf)]
  [[code](http://www.hamedkiani.com/uploads/5/1/8/8/51882963/bacf_toupload.zip)]
  [[project](http://www.hamedkiani.com/bacf.html)]

* **TSN:** Zhu Teng, Junliang Xing, Qiang Wang, Congyan Lang, Songhe Feng and Yi Jin.
  "Robust Object Tracking based on Temporal and Spatial Deep Networks." ICCV (2017).
  [[paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Teng_Robust_Object_Tracking_ICCV_2017_paper.pdf)]

* **p-tracker:** James Supančič, III; Deva Ramanan.
  "Tracking as Online Decision-Making: Learning a Policy From Streaming Videos With Reinforcement Learning." ICCV (2017).
  [[paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Supancic_Tracking_as_Online_ICCV_2017_paper.pdf)]
  [[supp](http://openaccess.thecvf.com/content_ICCV_2017/supplemental/Supancic_Tracking_as_Online_ICCV_2017_supplemental.pdf)]

* **DSiam:** Qing Guo; Wei Feng; Ce Zhou; Rui Huang; Liang Wan; Song Wang.
  "Learning Dynamic Siamese Network for Visual Object Tracking." ICCV (2017).
  [[paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Guo_Learning_Dynamic_Siamese_ICCV_2017_paper.pdf)]
  [[github](https://github.com/tsingqguo/DSiam)]

* **SP-KCF:** Xin Sun; Ngai-Man Cheung; Hongxun Yao; Yiluan Guo.
  "Non-Rigid Object Tracking via Deformable Patches Using Shape-Preserved KCF and Level Sets." ICCV (2017).
  [[paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Sun_Non-Rigid_Object_Tracking_ICCV_2017_paper.pdf)]

* **UCT:** Zheng Zhu, Guan Huang, Wei Zou, Dalong Du, Chang Huang.
  "UCT: Learning Unified Convolutional Networks for Real-Time Visual Tracking." ICCV workshop (2017).
  [[paper](http://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w28/Zhu_UCT_Learning_Unified_ICCV_2017_paper.pdf)]

* Tobias Bottger, Patrick Follmann.
  "The Benefits of Evaluating Tracker Performance Using Pixel-Wise Segmentations." ICCV workshop (2017).
  [[paper](http://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w28/Bottger_The_Benefits_of_ICCV_2017_paper.pdf)]

* **CFWCR:** Zhiqun He, Yingruo Fan, Junfei Zhuang, Yuan Dong, HongLiang Bai.
  "Correlation Filters With Weighted Convolution Responses." ICCV workshop (2017).
  [[paper](http://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w28/He_Correlation_Filters_With_ICCV_2017_paper.pdf)]
  [[github](https://github.com/he010103/CFWCR)]

* **IBCCF:** Feng Li, Yingjie Yao, Peihua Li, David Zhang, Wangmeng Zuo, Ming-Hsuan Yang.
  "Integrating Boundary and Center Correlation Filters for Visual Tracking With Aspect Ratio Variation." ICCV workshop (2017).
  [[paper](http://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w28/Li_Integrating_Boundary_and_ICCV_2017_paper.pdf)]
  [[github](https://github.com/lifeng9472/IBCCF)]

* **RFL:** Tianyu Yang, Antoni B. Chan.
  "Recurrent Filter Learning for Visual Tracking." ICCV workshop (2017).
  [[paper](http://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w28/Yang_Recurrent_Filter_Learning_ICCV_2017_paper.pdf)]


### CVPR2017

* **ECO:** Martin Danelljan, Goutam Bhat, Fahad Shahbaz Khan, Michael Felsberg. 
  "ECO: Efficient Convolution Operators for Tracking." CVPR (2017). 
  [[paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Danelljan_ECO_Efficient_Convolution_CVPR_2017_paper.pdf)]
  [[supp](http://openaccess.thecvf.com/content_cvpr_2017/supplemental/Danelljan_ECO_Efficient_Convolution_2017_CVPR_supplemental.pdf)]
  [[project](http://www.cvl.isy.liu.se/research/objrec/visualtracking/ecotrack/index.html)]
  [[github](https://github.com/martin-danelljan/ECO)]

* **CFNet:** Jack Valmadre, Luca Bertinetto, João F. Henriques, Andrea Vedaldi, Philip H. S. Torr.
  "End-to-end representation learning for Correlation Filter based tracking." CVPR (2017). 
  [[paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Valmadre_End-To-End_Representation_Learning_CVPR_2017_paper.pdf)]
  [[supp](http://openaccess.thecvf.com/content_cvpr_2017/supplemental/Valmadre_End-To-End_Representation_Learning_2017_CVPR_supplemental.pdf)]
  [[project](http://www.robots.ox.ac.uk/~luca/cfnet.html)]
  [[github](https://github.com/bertinetto/cfnet)]

* **CACF:** Matthias Mueller, Neil Smith, Bernard Ghanem. 
  "Context-Aware Correlation Filter Tracking." CVPR (2017 **oral**). 
  [[paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Mueller_Context-Aware_Correlation_Filter_CVPR_2017_paper.pdf)]
  [[supp](http://openaccess.thecvf.com/content_cvpr_2017/supplemental/Mueller_Context-Aware_Correlation_Filter_2017_CVPR_supplemental.zip)]
  [[project](https://ivul.kaust.edu.sa/Pages/pub-ca-cf-tracking.aspx)]
  [[code](https://github.com/thias15/Context-Aware-CF-Tracking)]

* **RaF:** Le Zhang, Jagannadan Varadarajan, Ponnuthurai Nagaratnam Suganthan, Narendra Ahuja and Pierre Moulin
  "Robust Visual Tracking Using Oblique Random Forests." CVPR (2017). 
  [[paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhang_Robust_Visual_Tracking_CVPR_2017_paper.pdf)]
  [[supp](http://openaccess.thecvf.com/content_cvpr_2017/supplemental/Zhang_Robust_Visual_Tracking_2017_CVPR_supplemental.pdf)]
  [[project](https://sites.google.com/site/zhangleuestc/incremental-oblique-random-forest)]
  [[code](https://github.com/ZhangLeUestc/Incremental-Oblique-Random-Forest)]

* **MCPF:** Tianzhu Zhang, Changsheng Xu, Ming-Hsuan Yang. 
  "Multi-Task Correlation Particle Filter for Robust Object Tracking." CVPR (2017). 
  [[paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhang_Multi-Task_Correlation_Particle_CVPR_2017_paper.pdf)]
  [[project](http://nlpr-web.ia.ac.cn/mmc/homepage/tzzhang/mcpf.html)]
  [[code](http://nlpr-web.ia.ac.cn/mmc/homepage/tzzhang/mcpf.html)]

* **ACFN:** Jongwon Choi, Hyung Jin Chang, Sangdoo Yun, Tobias Fischer, Yiannis Demiris, and Jin Young Choi.
  "Attentional Correlation Filter Network for Adaptive Visual Tracking." CVPR (2017).
  [[paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Choi_Attentional_Correlation_Filter_CVPR_2017_paper.pdf)]
  [[supp](http://openaccess.thecvf.com/content_cvpr_2017/supplemental/Choi_Attentional_Correlation_Filter_2017_CVPR_supplemental.pdf)]
  [[project](https://sites.google.com/site/jwchoivision/home/acfn-1)]
  [[test code](https://drive.google.com/file/d/0B0ZkG8zaRQoLQUswbW9qSWFaU0U/view?usp=drive_web)]
  [[training code](https://drive.google.com/file/d/0B0ZkG8zaRQoLZVVranBnbHlydnM/view?usp=drive_web)]

* **LMCF:** Mengmeng Wang, Yong Liu, Zeyi Huang. 
  "Large Margin Object Tracking with Circulant Feature Maps." CVPR (2017). 
  [[paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Wang_Large_Margin_Object_CVPR_2017_paper.pdf)]
  [[zhihu](https://zhuanlan.zhihu.com/p/25761718)]

* **ADNet:** Sangdoo Yun, Jongwon Choi, Youngjoon Yoo, Kimin Yun, Jin Young Choi.
  "Action-Decision Networks for Visual Tracking with Deep Reinforcement Learning." CVPR (2017 **Spotlight**). 
  [[paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Yun_Action-Decision_Networks_for_CVPR_2017_paper.pdf)]
  [[supp](http://openaccess.thecvf.com/content_cvpr_2017/supplemental/Yun_Action-Decision_Networks_for_2017_CVPR_supplemental.pdf)]
  [[project](https://sites.google.com/view/cvpr2017-adnet)]

* **CSR-DCF:** Alan Lukežič, Tomáš Vojíř, Luka Čehovin, Jiří Matas, Matej Kristan. 
  "Discriminative Correlation Filter with Channel and Spatial Reliability." CVPR (2017). 
  [[paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Lukezic_Discriminative_Correlation_Filter_CVPR_2017_paper.pdf)]
  [[supp](http://openaccess.thecvf.com/content_cvpr_2017/supplemental/Lukezic_Discriminative_Correlation_Filter_2017_CVPR_supplemental.pdf)]
  [[code](https://github.com/alanlukezic/csr-dcf)]

* **BranchOut:** Bohyung Han, Jack Sim, Hartwig Adam.
  "BranchOut: Regularization for Online Ensemble Tracking with Convolutional Neural Networks." CVPR (2017). 
  [[paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Han_BranchOut_Regularization_for_CVPR_2017_paper.pdf)]

* **AMCT:** Donghun Yeo, Jeany Son, Bohyung Han, Joonhee Han.
  "Superpixel-based Tracking-by-Segmentation using Markov Chains." CVPR (2017).
  [[paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Yeo_Superpixel-Based_Tracking-By-Segmentation_Using_CVPR_2017_paper.pdf)]

* **SANet:** Heng Fan, Haibin Ling. 
  "SANet: Structure-Aware Network for Visual Tracking." CVPRW (2017). 
  [[paper](https://arxiv.org/pdf/1611.06878.pdf)]
  [[project](http://www.dabi.temple.edu/~hbling/code/SANet/SANet.html)]
  [[code](http://www.dabi.temple.edu/~hbling/code/SANet/sanet_code.zip)]

### ECCV2016

* **SiameseFC:** Luca Bertinetto, Jack Valmadre, João F. Henriques, Andrea Vedaldi, Philip H.S. Torr. 
  "Fully-Convolutional Siamese Networks for Object Tracking." ECCV workshop (2016). 
  [[paper](http://120.52.73.78/arxiv.org/pdf/1606.09549v2.pdf)]
  [[project](http://www.robots.ox.ac.uk/~luca/siamese-fc.html)]
  [[github](https://github.com/bertinetto/siamese-fc)]

* **GOTURN:** David Held, Sebastian Thrun, Silvio Savarese. 
  "Learning to Track at 100 FPS with Deep Regression Networks." ECCV (2016). 
  [[paper](http://davheld.github.io/GOTURN/GOTURN.pdf)]
  [[project](http://davheld.github.io/GOTURN/GOTURN.html)]
  [[github](https://github.com/davheld/GOTURN)]

* **C-COT:** Martin Danelljan, Andreas Robinson, Fahad Khan, Michael Felsberg. 
  "Beyond Correlation Filters: Learning Continuous Convolution Operators for Visual Tracking." ECCV (2016). 
  [[paper](http://www.cvl.isy.liu.se/research/objrec/visualtracking/conttrack/C-COT_ECCV16.pdf)]
  [[project](http://www.cvl.isy.liu.se/research/objrec/visualtracking/conttrack/index.html)]
  [[github](https://github.com/martin-danelljan/Continuous-ConvOp)]

* **CF+AT:** Adel Bibi, Matthias Mueller, and Bernard Ghanem. 
  "Target Response Adaptation for Correlation Filter Tracking." ECCV (2016). 
  [[paper](http://www.adelbibi.com/papers/ECCV2016/Target_Adap.pdf)]
  [[project](https://ivul.kaust.edu.sa/Pages/pub-target-response-adaptation.aspx)]
  [[github](https://github.com/adelbibi/Target-Response-Adaptation-for-Correlation-Filter-Tracking)]

* Yao Sui, Ziming Zhang,  Guanghui Wang, Yafei Tang, Li Zhang. 
  "Real-Time Visual Tracking: Promoting the Robustness of Correlation Filter Learning." ECCV (2016). 
  [[paper](http://120.52.73.78/arxiv.org/pdf/1608.08173.pdf)]

* Yao Sui, Guanghui Wang, Yafei Tang, Li Zhang. 
  "Tracking Completion." ECCV (2016). 
  [[paper](http://120.52.73.78/arxiv.org/pdf/1608.08171v1.pdf)]

### CVPR2016

* **MDNet:** Nam, Hyeonseob, and Bohyung Han. 
  "Learning Multi-Domain Convolutional Neural Networks for Visual Tracking." CVPR (2016).
  [[paper](http://arxiv.org/pdf/1510.07945v2.pdf)]
  [[VOT_presentation](http://votchallenge.net/vot2015/download/presentation_Hyeonseob.pdf)]
  [[project](http://cvlab.postech.ac.kr/research/mdnet/)]
  [[github](https://github.com/HyeonseobNam/MDNet)]

* **SINT:** Ran Tao, Efstratios Gavves, Arnold W.M. Smeulders. 
  "Siamese Instance Search for Tracking." CVPR (2016).
  [[paper](https://staff.science.uva.nl/r.tao/pub/TaoCVPR2016.pdf)]
  [[project](https://staff.fnwi.uva.nl/r.tao/projects/SINT/SINT_proj.html)]

* **SCT:** Jongwon Choi, Hyung Jin Chang, Jiyeoup Jeong, Yiannis Demiris, and Jin Young Choi.
  "Visual Tracking Using Attention-Modulated Disintegration and Integration." CVPR (2016).
  [[paper](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Choi_Visual_Tracking_Using_CVPR_2016_paper.pdf)]
  [[project](https://sites.google.com/site/jwchoivision/home/sct)]

* **STCT:** Lijun Wang, Wanli Ouyang, Xiaogang Wang, and Huchuan Lu.
  "STCT: Sequentially Training Convolutional Networks for Visual Tracking." CVPR (2016).
  [[paper](http://www.ee.cuhk.edu.hk/~wlouyang/Papers/WangLJ_CVPR16.pdf)]
  [[github](https://github.com/scott89/STCT)]

* **SRDCFdecon:** Martin Danelljan, Gustav Häger, Fahad Khan, Michael Felsberg. 
  "Adaptive Decontamination of the Training Set: A Unified Formulation for Discriminative Visual Tracking." CVPR (2016).
  [[paper](https://www.cvl.isy.liu.se/research/objrec/visualtracking/decontrack/AdaptiveDecon_CVPR16.pdf)]
  [[project](https://www.cvl.isy.liu.se/research/objrec/visualtracking/decontrack/index.html)]

* **HDT:** Yuankai Qi, Shengping Zhang, Lei Qin, Hongxun Yao, Qingming Huang, Jongwoo Lim, Ming-Hsuan Yang. 
  "Hedged Deep Tracking." CVPR (2016). 
  [[paper](http://faculty.ucmerced.edu/mhyang/papers/cvpr16_hedge_tracking.pdf)]
  [[project](https://sites.google.com/site/yuankiqi/hdt/)]

* **Staple:** Luca Bertinetto, Jack Valmadre, Stuart Golodetz, Ondrej Miksik, Philip H.S. Torr. 
  "Staple: Complementary Learners for Real-Time Tracking." CVPR (2016). 
  [[paper](http://120.52.73.75/arxiv.org/pdf/1512.01355v2.pdf)]
  [[project](http://www.robots.ox.ac.uk/~luca/staple.html)]
  [[github](https://github.com/bertinetto/staple)]

* **EBT:** Gao Zhu, Fatih Porikli, and Hongdong Li.
  "Beyond Local Search: Tracking Objects Everywhere with Instance-Specific Proposals." CVPR (2016). 
  [[paper](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Zhu_Beyond_Local_Search_CVPR_2016_paper.pdf)]
  [[exe](http://www.votchallenge.net/vot2016/download/02_EBT.zip)]

* **DLSSVM:** Jifeng Ning, Jimei Yang, Shaojie Jiang, Lei Zhang and Ming-Hsuan Yang. 
  "Object Tracking via Dual Linear Structured SVM and Explicit Feature Map." CVPR (2016). 
  [[paper](http://www4.comp.polyu.edu.hk/~cslzhang/paper/cvpr16/DLSSVM.pdf)]
  [[code](http://www4.comp.polyu.edu.hk/~cslzhang/code/DLSSVM_CVPR.zip)]
  [[project](http://www4.comp.polyu.edu.hk/~cslzhang/DLSSVM/DLSSVM.htm)]

### NIPS2016
* **Learnet:** Luca Bertinetto, João F. Henriques, Jack Valmadre, Philip H. S. Torr, Andrea Vedaldi. 
  "Learning feed-forward one-shot learners." NIPS (2016). 
  [[paper](https://arxiv.org/pdf/1606.05233v1.pdf)]

### ICCV2015

* **FCNT:** Lijun Wang, Wanli Ouyang, Xiaogang Wang, and Huchuan Lu. 
  "Visual Tracking with Fully Convolutional Networks." ICCV (2015). 
  [[paper](http://202.118.75.4/lu/Paper/ICCV2015/iccv15_lijun.pdf)]
  [[project](http://scott89.github.io/FCNT/)]
  [[github](https://github.com/scott89/FCNT)]

* **SRDCF:** Martin Danelljan, Gustav Häger, Fahad Khan, Michael Felsberg. 
  "Learning Spatially Regularized Correlation Filters for Visual Tracking." ICCV (2015). 
  [[paper](https://www.cvl.isy.liu.se/research/objrec/visualtracking/regvistrack/SRDCF_ICCV15.pdf)]
  [[project](https://www.cvl.isy.liu.se/research/objrec/visualtracking/regvistrack/)]

* **CF2:** Chao Ma, Jia-Bin Huang, Xiaokang Yang and Ming-Hsuan Yang.
  "Hierarchical Convolutional Features for Visual Tracking." ICCV (2015)
  [[paper](http://faculty.ucmerced.edu/mhyang/papers/iccv15_tracking.pdf)]
  [[project](https://sites.google.com/site/jbhuang0604/publications/cf2)]
  [[github](https://github.com/jbhuang0604/CF2)]

* Naiyan Wang, Jianping Shi, Dit-Yan Yeung and Jiaya Jia.
  "Understanding and Diagnosing Visual Tracking Systems." ICCV (2015). 
  [[paper](http://winsty.net/papers/diagnose.pdf)]
  [[project](http://winsty.net/tracker_diagnose.html)]
  [[code](http://winsty.net/diagnose/diagnose_code.zip)]\

* **DeepSRDCF:** Martin Danelljan, Gustav Häger, Fahad Khan, Michael Felsberg. 
  "Convolutional Features for Correlation Filter Based Visual Tracking." ICCV workshop (2015). 
  [[paper](https://www.cvl.isy.liu.se/research/objrec/visualtracking/regvistrack/ConvDCF_ICCV15_VOTworkshop.pdf)]
  [[project](https://www.cvl.isy.liu.se/research/objrec/visualtracking/regvistrack/)]

* **RAJSSC:** Mengdan Zhang, Junliang Xing, Jin Gao, Xinchu Shi, Qiang Wang, Weiming Hu. 
  "Joint Scale-Spatial Correlation Tracking with Adaptive Rotation Estimation." ICCV workshop (2015). 
  [[paper](http://www.cv-foundation.org//openaccess/content_iccv_2015_workshops/w14/papers/Zhang_Joint_Scale-Spatial_Correlation_ICCV_2015_paper.pdf)]
  [[poster](http://www.votchallenge.net/vot2015/download/poster_Mengdan_Zhang.pdf)]

### CVPR2015

* **MUSTer:** Zhibin Hong, Zhe Chen, Chaohui Wang, Xue Mei, Danil Prokhorov, Dacheng Tao. 
  "MUlti-Store Tracker (MUSTer): A Cognitive Psychology Inspired Approach to Object Tracking." CVPR (2015). 
  [[paper](http://openaccess.thecvf.com/content_cvpr_2015/papers/Hong_MUlti-Store_Tracker_MUSTer_2015_CVPR_paper.pdf)]
  [[project](https://sites.google.com/site/multistoretrackermuster/)]

* **LCT:** Chao Ma, Xiaokang Yang, Chongyang Zhang, Ming-Hsuan Yang.
  "Long-term Correlation Tracking." CVPR (2015).
  [[paper](http://openaccess.thecvf.com/content_cvpr_2015/papers/Ma_Long-Term_Correlation_Tracking_2015_CVPR_paper.pdf)]
  [[project](https://sites.google.com/site/chaoma99/cvpr15_tracking)]
  [[github](https://github.com/chaoma99/lct-tracker)]

* **DAT:** Horst Possegger, Thomas Mauthner, and Horst Bischof. 
  "In Defense of Color-based Model-free Tracking." CVPR (2015). 
  [[paper](https://lrs.icg.tugraz.at/pubs/possegger_cvpr15.pdf)]
  [[project](https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/dat)]
  [[code](https://lrs.icg.tugraz.at/downloads/dat-v1.0.zip)]

* **RPT:** Yang Li, Jianke Zhu and Steven C.H. Hoi. 
  "Reliable Patch Trackers: Robust Visual Tracking by Exploiting Reliable Patches." CVPR (2015). 
  [[paper](https://github.com/ihpdep/ihpdep.github.io/raw/master/papers/cvpr15_rpt.pdf)]
  [[github](https://github.com/ihpdep/rpt)]

### ICML2015

* **CNN-SVM:** Seunghoon Hong, Tackgeun You, Suha Kwak and Bohyung Han.
  "Online Tracking by Learning Discriminative Saliency Map with Convolutional Neural Network ." ICML (2015)
  [[paper](http://120.52.73.80/arxiv.org/pdf/1502.06796.pdf)]
  [[project](http://cvlab.postech.ac.kr/research/CNN_SVM/)]

### BMVC2014

* **DSST:** Martin Danelljan, Gustav Häger, Fahad Shahbaz Khan and Michael Felsberg. 
  "Accurate Scale Estimation for Robust Visual Tracking." BMVC (2014).
  [[paper](http://www.cvl.isy.liu.se/research/objrec/visualtracking/scalvistrack/ScaleTracking_BMVC14.pdf)]
  [[PAMI](http://www.cvl.isy.liu.se/en/research/objrec/visualtracking/scalvistrack/DSST_TPAMI.pdf)]
  [[project](http://www.cvl.isy.liu.se/en/research/objrec/visualtracking/scalvistrack/index.html)]

### ECCV2014

* **MEEM:** Jianming Zhang, Shugao Ma, and Stan Sclaroff.
  "MEEM: Robust Tracking via Multiple Experts using Entropy Minimization." ECCV (2014).
  [[paper](http://cs-people.bu.edu/jmzhang/MEEM/MEEM-eccv-preprint.pdf)]
  [[project](http://cs-people.bu.edu/jmzhang/MEEM/MEEM.html)]

* **TGPR:** Jin Gao, Haibin Ling, Weiming Hu, Junliang Xing.
  "Transfer Learning Based Visual Tracking with Gaussian Process Regression." ECCV (2014).
  [[paper](http://www.dabi.temple.edu/~hbling/publication/tgpr-eccv14.pdf)]
  [[project](http://www.dabi.temple.edu/~hbling/code/TGPR.htm)]

* **STC:** Kaihua Zhang, Lei Zhang, Ming-Hsuan Yang, David Zhang.
  "Fast Tracking via Spatio-Temporal Context Learning." ECCV (2014).
  [[paper](http://arxiv.org/pdf/1311.1939v1.pdf)]
  [[project](http://www4.comp.polyu.edu.hk/~cslzhang/STC/STC.htm)]

* **SAMF:** Yang Li, Jianke Zhu.
  "A Scale Adaptive Kernel Correlation Filter Tracker with Feature Integration." ECCV workshop (2014).
  [[paper](http://link.springer.com/content/pdf/10.1007%2F978-3-319-16181-5_18.pdf)]
  [[github](https://github.com/ihpdep/samf)]

### NIPS2013

* **DLT:** Naiyan Wang and Dit-Yan Yeung. 
  "Learning A Deep Compact Image Representation for Visual Tracking." NIPS (2013). 
  [[paper](http://winsty.net/papers/dlt.pdf)]
  [[project](http://winsty.net/dlt.html)]
  [[code](http://winsty.net/dlt/DLTcode.zip)]

### PAMI & IJCV & TIP

* **AOGTracker:** Tianfu Wu , Yang Lu and Song-Chun Zhu. 
  "Online Object Tracking, Learning and Parsing with And-Or Graphs." TPAMI (2017).
  [[paper](http://www4.ncsu.edu/~twu19/papers/AOGTracker_PAMI.pdf)]
  [[project](http://www4.ncsu.edu/~twu19/project_posts/AOGTracker/)]
  [[github](https://github.com/tfwu/RGM-AOGTracker)] 

* **MCPF:** Tianzhu Zhang, Changsheng Xu, Ming-Hsuan Yang.
    " Learning Multi-task Correlation Particle Filters for Visual Tracking." TPAMI (2017).
      [[paper]]
      [[project](http://nlpr-web.ia.ac.cn/mmc/homepage/tzzhang/lmcpf.html)]
      [[code](http://nlpr-web.ia.ac.cn/mmc/homepage/tzzhang/Project_Tianzhu/zhang_mcpf/Source_Code/Source_Code.zip)] 

* **RSST:** Tianzhu Zhang, Changsheng Xu, Ming-Hsuan Yang.
  " Robust Structural Sparse Tracking." TPAMI (2017).
  [[paper]]
  [[project](http://nlpr-web.ia.ac.cn/mmc/homepage/tzzhang/rsst.html)]
  [[code](http://nlpr-web.ia.ac.cn/mmc/homepage/tzzhang/Project_Tianzhu/zhang_RSST/RSSTDeep/RSSTDeep_Code.zip)] 

* **fDSST:** Martin Danelljan, Gustav Häger, Fahad Khan, Michael Felsberg.
  "Discriminative Scale Space Tracking." TPAMI (2017).
  [[paper](http://www.cvl.isy.liu.se/research/objrec/visualtracking/scalvistrack/DSST_TPAMI.pdf)]
  [[project](http://www.cvl.isy.liu.se/research/objrec/visualtracking/scalvistrack/index.html)]
  [[code](http://www.cvl.isy.liu.se/research/objrec/visualtracking/scalvistrack/fDSST_code.zip)] 

* **KCF:** João F. Henriques, Rui Caseiro, Pedro Martins, Jorge Batista. 
  "High-Speed Tracking with Kernelized Correlation Filters." TPAMI (2015).
  [[paper](http://www.robots.ox.ac.uk/~joao/publications/henriques_tpami2015.pdf)]
  [[project](http://www.robots.ox.ac.uk/~joao/circulant/)]

* **CLRST:** Tianzhu Zhang, Si Liu, Narendra Ahuja, Ming-Hsuan Yang, Bernard Ghanem.  
  "Robust Visual Tracking Via Consistent Low-Rank Sparse Learning." IJCV (2015). 
  [[paper](http://nlpr-web.ia.ac.cn/mmc/homepage/tzzhang/tianzhu%20zhang_files/Journal%20Articles/IJCV15_zhang_Low-Rank%20Sparse%20Learning.pdf)]
  [[project](http://nlpr-web.ia.ac.cn/mmc/homepage/tzzhang/Project_Tianzhu/zhang_IJCV14/Robust%20Visual%20Tracking%20Via%20Consistent%20Low-Rank%20Sparse.html)]
  [[code](http://nlpr-web.ia.ac.cn/mmc/homepage/tzzhang/Project_Tianzhu/zhang_IJCV14/material/LRT_Code.zip)]

* **DNT:** Zhizhen Chi, Hongyang Li, Huchuan Lu, Ming-Hsuan Yang. 
  "Dual Deep Network for Visual Tracking." TIP (2017). 
  [[paper](https://arxiv.org/pdf/1612.06053v1.pdf)]

* **DRT:** Junyu Gao, Tianzhu Zhang, Xiaoshan Yang, Changsheng Xu. 
  "Deep Relative Tracking." TIP (2017). 
  [[paper](http://ieeexplore.ieee.org/abstract/document/7828108/)]

* **BIT:** Bolun Cai, Xiangmin Xu, Xiaofen Xing, Kui Jia, Jie Miao, Dacheng Tao.
  "BIT: Biologically Inspired Tracker." TIP (2016). 
  [[paper](http://caibolun.github.io/papers/BIT_TIP.pdf)]
  [[project](http://caibolun.github.io/BIT/index.html)]
  [[github](https://github.com/caibolun/BIT)]

* **CNT:** Kaihua Zhang, Qingshan Liu, Yi Wu, Minghsuan Yang. 
  "Robust Visual Tracking via Convolutional Networks Without Training." TIP (2016). 
  [[paper](http://kaihuazhang.net/CNT.pdf)]
  [[code](http://kaihuazhang.net/CNT_matlab.rar)]

## ArXiv

* **MLT:** Janghoon Choi, Junseok Kwon, Kyoung Mu Lee.
  "Deep Meta Learning for Real-Time Visual Tracking based on Target-Specific Feature Space." arXiv (2017). 
  [[paper](https://arxiv.org/pdf/1712.09153v1.pdf)]

* **STECF:** Yang Li, Jianke Zhu, Wenjie Song, Zhefeng Wang, Hantang Liu, Steven C. H. Hoi.
  "Robust Estimation of Similarity Transformation for Visual Object Tracking with Correlation Filters." arXiv (2017). 
  [[paper](https://arxiv.org/pdf/1712.05231v1.pdf)]

* **PAWSS:** Xiaofei Du, Alessio Dore, Danail Stoyanov. 
  "Patch-based adaptive weighting with segmentation and scale (PAWSS) for visual tracking." arXiv (2017). 
  [[paper](https://arxiv.org/pdf/1708.01179v1.pdf)]

* **SFT:** Zhen Cui, You yi Cai, Wen ming Zheng, Jian Yang. 
  "Spectral Filter Tracking." arXiv (2017). 
  [[paper](https://arxiv.org/pdf/1707.05553v1.pdf)]

* **Re3:** Daniel Gordon, Ali Farhadi, Dieter Fox. 
  "Re3 : Real-Time Recurrent Regression Networks for Object Tracking." arXiv (2017). 
  [[paper](https://arxiv.org/pdf/1705.06368.pdf)]

* **DCFNet:** Qiang Wang, Jin Gao, Junliang Xing, Mengdan Zhang, Weiming Hu. 
  "DCFNet: Discriminant Correlation Filters Network for Visual Tracking." arXiv (2017). 
  [[paper](https://arxiv.org/pdf/1704.04057.pdf)]
  [[code](https://github.com/foolwood/DCFNet#dcfnet-discriminant-correlation-filters-network-for-visual-tracking)]

* **TCNN:** Hyeonseob Nam, Mooyeol Baek, Bohyung Han. 
  "Modeling and Propagating CNNs in a Tree Structure for Visual Tracking." arXiv (2016). 
  [[paper](http://arxiv.org/pdf/1608.07242v1.pdf)]
  [[code](http://www.votchallenge.net/vot2016/download/44_TCNN.zip)]

* **RDT:** Janghoon Choi, Junseok Kwon, Kyoung Mu Lee. 
  "Visual Tracking by Reinforced Decision Making." arXiv (2017). 
  [[paper](https://arxiv.org/pdf/1702.06291.pdf)]

* **MSDAT:** Xinyu Wang, Hanxi Li, Yi Li, Fumin Shen, Fatih Porikli .
  "Robust and Real-time Deep Tracking Via Multi-Scale Domain Adaptation." arXiv (2017). 
  [[paper](https://arxiv.org/pdf/1701.00561.pdf)]

* **RLT:** Da Zhang, Hamid Maei, Xin Wang, Yuan-Fang Wang.
  "Deep Reinforcement Learning for Visual Object Tracking in Videos." arXiv (2017). 
  [[paper](https://arxiv.org/pdf/1701.08936v1.pdf)]

* **SCF:** Wangmeng Zuo, Xiaohe Wu, Liang Lin, Lei Zhang, Ming-Hsuan Yang. 
  "Learning Support Correlation Filters for Visual Tracking." arXiv (2016).
  [[paper](https://arxiv.org/pdf/1601.06032.pdf)]
  [[project](http://faculty.ucmerced.edu/mhyang/project/scf/)]

* **CRT:** Kai Chen, Wenbing Tao. 
  "Convolutional Regression for Visual Tracking." arXiv (2016). 
  [[paper](https://arxiv.org/pdf/1611.04215.pdf)]

* **BMR:** Kaihua Zhang, Qingshan Liu, and Ming-Hsuan Yang. 
  "Visual Tracking via Boolean Map Representations." arXiv (2016). 
  [[paper](https://arxiv.org/pdf/1610.09652v1.pdf)]

* **YCNN:** Kai Chen, Wenbing Tao. 
  "Once for All: a Two-flow Convolutional Neural Network for Visual Tracking." arXiv (2016). 
  [[paper](https://arxiv.org/pdf/1604.07507v1.pdf)]

* **ROLO:** Guanghan Ning, Zhi Zhang, Chen Huang, Zhihai He, Xiaobo Ren, Haohong Wang. 
  "Spatially Supervised Recurrent Convolutional Neural Networks for Visual Object Tracking." arXiv (2016). 
  [[paper](http://arxiv.org/pdf/1607.05781v1.pdf)]
  [[project](http://guanghan.info/projects/ROLO/)]
  [[github](https://github.com/Guanghan/ROLO/)]

* **RATM:** Samira Ebrahimi Kahou, Vincent Michalski, Roland Memisevic. 
  "RATM: Recurrent Attentive Tracking Model." arXiv (2015). 
  [[paper](https://arxiv.org/pdf/1510.08660v4.pdf)]
  [[github](https://github.com/saebrahimi/RATM)]

* **SO-DLT:** Naiyan Wang, Siyi Li, Abhinav Gupta, Dit-Yan Yeung. 
  "Transferring Rich Feature Hierarchies for Robust Visual Tracking." arXiv (2015). 
  [[paper](https://arxiv.org/pdf/1501.04587v2.pdf)]
  [[code](http://www.votchallenge.net/vot2016/download/08_SO-DLT.zip)]

* **DMSRDCF:** Susanna Gladh, Martin Danelljan, Fahad Shahbaz Khan, Michael Felsberg. 
  "Deep Motion Features for Visual Tracking." ICPR **Best Paper** (2016). 
  [[paper](https://arxiv.org/pdf/1612.06615v1.pdf)]

    ### **Multi-Modal**
    
    <!-- RGB Depth -->  
    - [**RGB-D**](#RGB-D)
        - [**2021**](#2021)
            * **DeT**: Song Yan, Jinyu Yang, Jani Käpylä, Feng Zheng, Aleš Leonardis, Joni-Kristian Kämäräinen. DepthTrack : Unveiling the Power of RGBD Tracking. In _ICCV_, 2021. [[Paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Yan_DepthTrack_Unveiling_the_Power_of_RGBD_Tracking_ICCV_2021_paper.pdf)
            * **TSDM**: Pengyao Zhao, Quanli Liu, Wei Wang and Qiang Guo. TSDM: Tracking by SiamRPN++ with a Depth-refiner and a Mask-generator. In _ICPR_, 2021. [[Paper]](https://arxiv.org/ftp/arxiv/papers/2005/2005.04063.pdf) [[Code]](https://github.com/lql-team/TSDM)
            * **3s-RGBD**: Feng Xiao, Qiuxia Wu, Han Huang. Single-scale siamese network based RGB-D object tracking with adaptive bounding boxes. In _Neurocomputing_, 2021. [[Paper]](https://www.sciencedirect.com/sdfe/reader/pii/S0925231221005439/pdf)
        - [**2020**](#2020)
            * **DAL**: Yanlin Qian, Alan Lukezic, Matej Kristan, Joni-Kristian Kämäräinen, Jiri Matas. DAL : A deep depth-aware long-term tracker. In _ICPR_, 2020. [[Code]](https://github.com/xiaozai/DAL)
            * **RF-CFF**: Yong Wang, Xian Wei, Hao Shen, Lu Ding, Jiuqing Wan. Robust fusion for RGB-D tracking using CNN features. In _Applied Soft Computing Journal_, 2020. [[Paper]](https://www.sciencedirect.com/sdfe/reader/pii/S1568494620302428/pdf)
            * **SiamOC**: Wenli Zhang, Kun Yang, Yitao Xin, Rui Meng. An Occlusion-Aware RGB-D Visual Object Tracking Method Based on Siamese Network. In _ICSP_, 2020. [[Paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9320907)
            * **WCO**: Weichun Liu, Xiaoan Tang, Chengling Zhao. Robust RGBD Tracking via Weighted Convlution Operators. In _Sensors_ 20(8), 2020. [[Paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8950173/)
        - [**2019**](#2019)
            * **3DMS**: Alexander Gutev, Carl James Debono. Exploiting Depth Information to Increase Object Tracking Robustness. In _ICST_ 2019. [[Paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8861628/)
            * **CA3DMS**: Ye Liu, Xiao-Yuan Jing, Jianhui Nie, Hao Gao, Jun Liu, Guo-Ping Jiang. Context-Aware Three-Dimensional Mean-Shift With Occlusion Handling for Robust Object Tracking in RGB-D Videos. In _TMM_ 21(3), 2019. [[Paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8425768) [[Code]](https://github.com/yeliu2013/ca3dms-toh)
            * **Depth-CCF**: Guanqun Li, Lei Huang, Peichang Zhang, Qiang Li, YongKai Huo. Depth Information Aided Constrained correlation Filter for Visual Tracking. In _GSKI_ 2018. [[Paper]](https://iopscience.iop.org/article/10.1088/1755-1315/234/1/012005)
            * **ECO_TA**: Yangliu Kuai, Gongjian Wen, Dongdong Li, Jingjing Xiao. Target-Aware Correlation Filter Tracking in RGBD Videos. In _Sensors_ 19(20), 2019. [[Paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8752050)
            * **H-FCN**: Ming-xin Jiang, Chao Deng, Jing-song Shan, Yuan-yuan Wang, Yin-jie Jia, Xing Sun. Hierarchical multi-modal fusion FCN with attention model for RGB-D tracking. In _Information Fusion_ [[Paper]](https://www.sciencedirect.com/sdfe/reader/pii/S1566253517306784/pdf)
            * **OTR**: Ugur Kart, Alan Lukezic, Matej Kristan, Joni-Kristian Kamarainen, Jiri Matas. Object Tracking by Reconstruction with View-Specific Discriminative Correlation Filters. In _CVPR_ 2019. [[Paper]](https://openaccess.thecvf.com/content_CVPR_2019/papers/Kart_Object_Tracking_by_Reconstruction_With_View-Specific_Discriminative_Correlation_Filters_CVPR_2019_paper.pdf) [[Code]](https://github.com/ugurkart/OTR)
            * **RGBD-OD**: Yujun Xie, Yao Lu, Shuang Gu. RGB-D Object Tracking with Occlusion Detection. In _CIS_ 2019. [[Paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9023755)
        - [**2018**](#2018)
            * **CSR-rgbd**: Uğur Kart, Joni-Kristian Kämäräinen, Jiří Matas. How to Make an RGBD Tracker? In _ECCV Workshop_ 2018. [[Paper]](https://link.springer.com/chapter/10.1007/978-3-030-11009-3_8) [[Code]](http://tracking.cs.princeton.edu/)
            * **DM-DCF**: Uğur Kart, Joni-Kristian Kämäräinen, Jiří Matas, Lixin Fan, Francesco Cricri. Depth Masked Discriminative Correlation Filter. In _ICPR_ 2018. [[Paper]](https://arxiv.org/pdf/1802.09227.pdf)
            * **OACPF**: Yayu Zhai, Ping Song, Zonglei Mou, Xiaoxiao Chen, Xiongjun Liu. Occlusion-Aware Correlation Particle FilterTarget Tracking Based on RGBD Data. In _Access_ (6), 2018. [[Paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8463446)
            * **RT-KCF**: Han Zhang, Meng Cai, Jianxun Li.  A Real-time RGB-D tracker based on KCF. In _CCDC_ 2018. [[Paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8407972)
            * **SEOH**: Jiaxu Leng, Ying Liu. Real-Time RGB-D Visual Tracking With ScaleEstimation and Occlusion Handling. In _Access_ (6), 2018. [[Paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8353501)
            * **STC**: Jingjing Xiao, Rustam Stolkin, Yuqing Gao, Aleš Leonardis. Robust Fusion of Color and Depth Data for RGB-D Target Tracking Using Adaptive Range-Invariant Depth Models and Spatio-Temporal Consistency Constraints. In _TC_ 48(8) 2018. [[Paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8026575) [[Code]](https://github.com/shine636363/RGBDtracker)
         - [**2017**](#2017)
              * **ODIOT**: Wei-Long Zheng, Shan-Chun Shen, Bao-Liang Lu. Online Depth Image-Based Object Tracking with Sparse Representation and Object Detection. In _Neural Process Letters_, 2017. [[Paper]](https://link.springer.com/content/pdf/10.1007/s11063-016-9509-y.pdf)
              * **ROTSL**: Zi-ang Ma, Zhi-yu Xiang. Robust Object Tracking with RGBD-based Sparse Learning. In _ITEE_ (18), 2017. [[Paper]](https://link.springer.com/article/10.1631/FITEE.1601338)
         - [**2016**](#2016)
              * **DLS**:  Ning An, Xiao-Guang Zhao, Zeng-Guang Hou.  Online RGB-D Tracking via Detection-Learning-Segmentation. In _ICPR_ 2016. [[Paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7899805)
              * **DS-KCF_shape**: Sion Hannuna, Massimo Camplani, Jake Hall, Majid Mirmehdi, Dima Damen, Tilo Burghardt, Adeline Paiement, Lili Tao. DS-KCF: A Real-time Tracker for RGB-D Data. In _RTIP_ (16), 2016. [[Paper]](https://link.springer.com/content/pdf/10.1007/s11554-016-0654-3.pdf) [[Code]](https://github.com/mcamplan/DSKCF_JRTIP2016)
              * **3D-T**: Adel Bibi, Tianzhu Zhang, Bernard Ghanem. 3D Part-Based Sparse Tracker with Automatic Synchronization and Registration. In _CVPR_ 2016. [[Paper]](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Bibi_3D_Part-Based_Sparse_CVPR_2016_paper.pdf) [[Code]](https://github.com/adelbibi/3D-Part-Based-Sparse-Tracker-with-Automatic-Synchronization-and-Registration)
              * **OAPF**: Kourosh Meshgia, Shin-ichi Maedaa, Shigeyuki Obaa, Henrik Skibbea, Yu-zhe Lia, Shin Ishii. Occlusion Aware Particle Filter Tracker to Handle Complex and Persistent Occlusions. In _CVIU_ (150), 2016. [[Paper]](http://ishiilab.jp/member/meshgi-k/files/ai/prl14/OAPF.pdf)
         - [**2015**](#2015)
              * **CDG**: Huizhang Shi, Changxin Gao, Nong Sang. Using Consistency of Depth Gradient to Improve Visual Tracking in RGB-D sequences. In _CAC_, 2015. [[Paper]](https://ieeexplore.ieee.org/document/7382555)
              * **DS-KCF**: Massimo Camplani, Sion Hannuna, Majid Mirmehdi, Dima Damen, Adeline Paiement, Lili Tao, Tilo Burghardt. Real-time RGB-D Tracking with Depth Scaling Kernelised Correlation Filters and Occlusion Handling. In _BMVC_, 2015. [[Paper]](https://core.ac.uk/reader/78861956) [[Code]](https://github.com/mcamplan/DSKCF_BMVC2015)
              * **DOHR**: Ping Ding, Yan Song. Robust Object Tracking Using Color and Depth Images with a Depth Based Occlusion Handling and Recovery. In _FSKD_, 2015. [[Paper]](https://ieeexplore.ieee.org/document/7382068)
              * **ISOD**: Yan Chen, Yingju Shen, Xin Liu, Bineng Zhong. 3D Object Tracking via Image Sets and Depth-Based Occlusion Detection. In _SP_ (112), 2015. [[Paper]](https://www.sciencedirect.com/science/article/pii/S0165168414004204)
              * **OL3DC**: Bineng Zhong, Yingju Shen, Yan Chen, Weibo Xie, Zhen Cui, Hongbo Zhang, Duansheng Chen ,Tian Wang, Xin Liu, Shujuan Peng, Jin Gou, Jixiang Du, Jing Wang, Wenming Zheng. Online Learning 3D Context for Robust Visual Tracking. In _Neurocomputing_ (151), 2015. [[Paper]](https://www.sciencedirect.com/science/article/pii/S0925231214013757)
         - [**2014**](#2014)
              * **MCBT**: Qi Wang, Jianwu Fang, Yuan Yuan. Multi-Cue Based Tracking. In _Neurocomputing_ (131), 2014. [[Paper]](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.700.8771&rep=rep1&type=pdf)
         - [**2013**](#2013)
              * **PT**: Shuran Song, Jianxiong Xiao. Tracking Revisited using RGBD Camera: Unified Benchmark and Baselines. In _ICCV_, 2013. [[Paper]](https://vision.princeton.edu/projects/2013/tracking/paper.pdf) [[Project]](https://tracking.cs.princeton.edu/index.html)
         - [**2012**](#2012)
              * **AMCT**: Germán Martín García, Dominik Alexander Klein, Jörg Stückler, Simone Frintrop, Armin B. Cremers. Adaptive Multi-cue 3D Tracking of Arbitrary Objects. In _JDOS_, 2012. [[Paper]](https://link.springer.com/chapter/10.1007/978-3-642-32717-9_36)
             
    <!-- RGB Nature Language -->   
    - [**RGB-NL**](#RGB-NL)
         - [**2021**](#2021)
              * **TNL2K**: Wang, Xiao and Shu, Xiujun and Zhang, Zhipeng and Jiang, Bo and Wang, Yaowei and Tian, Yonghong and Wu, Feng. Towards More Flexible and Accurate Object Tracking with Natural Language: Algorithms and Benchmark. In _CVPR_ 2021. [[Paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_Towards_More_Flexible_and_Accurate_Object_Tracking_With_Natural_Language_CVPR_2021_paper.pdf) [[Project]](https://sites.google.com/view/langtrackbenchmark/)
              * **SNLT**: Feng, Qi and Ablavsky, Vitaly and Bai, Qinxun and Sclaroff, Stan. Siamese Natural Language Tracker: Tracking by Natural Language Descriptions with Siamese Trackers. In _CVPR_ 2021. [[Paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Feng_Siamese_Natural_Language_Tracker_Tracking_by_Natural_Language_Descriptions_With_CVPR_2021_paper.pdf) [[Code]](https://github.com/fredfung007/snlt)
         - [**2020**](#2020)
              * **GTI**: Yang, Zhengyuan, Tushar Kumar, Tianlang Chen, Jingsong Su, and Jiebo Luo. Grounding-tracking-integration. In _TCSV_ 2020. [[Paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9261416)
         - [**2018**](#2018)
              * **DAT**: Wang, Xiao, Chenglong Li, Rui Yang, Tianzhu Zhang, Jin Tang, and Bin Luo. Describe and attend to track: Learning natural language guided structural representation and visual attention for object tracking. In _arXiv_ 2018. [[Paper]](https://arxiv.org/pdf/1811.10014.pdf)
         - [**2017**](#2017)
              * **TNLS**: Li, Zhenyang, Ran Tao, Efstratios Gavves, Cees GM Snoek, and Arnold WM Smeulders. "Tracking by natural language specification." In _CVPR_ 2017. [[Paper]](https://openaccess.thecvf.com/content_cvpr_2017/papers/Li_Tracking_by_Natural_CVPR_2017_paper.pdf)

    <!-- RGB EVENT -->
    - [**RGB-Event**](#RGB-Event)
         - [**Survey**](#Survey)
              * Gallego, Guillermo, Tobi Delbrück, Garrick Orchard, Chiara Bartolozzi, Brian Taba, Andrea Censi, Stefan Leutenegger et al. Event-based vision: A survey. In _TPAMI_ 2020. [[Paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9138762)
         - [**2022**](#2022)
              * **VisEvent**: Wang, Xiao, Jianing Li, Lin Zhu, Zhipeng Zhang, Zhe Chen, Xin Li, Yaowei Wang, Yonghong Tian, and Feng Wu. VisEvent: Reliable Object Tracking via Collaboration of Frame and Event Flows. In _AAAI_ 2022. [[Paper]](https://arxiv.org/pdf/2108.05015.pdf) [[Project]](https://sites.google.com/view/viseventtrack/)
         - [**2021**](#2021)
              * **JEFE**: Zhang, Jiqing, Xin Yang, Yingkai Fu, Xiaopeng Wei, Baocai Yin, and Bo Dong. Object Tracking by Jointly Exploiting Frame and Event Domain. In _ICCV_ 2021. [[Paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhang_Object_Tracking_by_Jointly_Exploiting_Frame_and_Event_Domain_ICCV_2021_paper.pdf)
              
    <!-- RGB Thermal -->   
    - [**RGB-T**](#RGB-T)
         - [**2022**](#2022)
            * **VTUAV**: Pengyu Zhang, Jie Zhao, Dong Wang, Huchuan Lu, Xiang Ruan. Visible-Thermal UAV Tracking: A Large-Scale Benchmark and New Baseline", In _CVPR_ 2022. [[Paper]](https://arxiv.org/abs/2204.04120) [[Project]](https://zhang-pengyu.github.io/DUT-VTUAV/)
         - [**2021**](#2021)
            * **CBPNet**: Qin Xu, Yiming Mei, Jinpei Liu, Chenglong Li. Multimodal Cross-Layer Bilinear Pooling for RGBT Tracking", In _TMM_ 2021. [[Paper]](https://ieeexplore.ieee.org/abstract/document/9340007?casa_token=2J66RsN_jAQAAAAA:f6O-OSYp3Hwco_zzTP7175Oq35qaFNRvgX29LqMfjfO7Ya4vAHCCkuNJabgtoOusYTaH89kB)

  - [Metrics](#metrics)
  - [Benchmark Results](#benchmark-results)
    - [VOT16](https://www.votchallenge.net/vot2016/)
    - [VOT17](https://www.votchallenge.net/vot2017/)
    - [VOT18](https://www.votchallenge.net/vot2018/)
    - [VOT19](https://www.votchallenge.net/vot2019/)
    - [VOT20](https://www.votchallenge.net/vot2020/)
    - [VOT21](https://www.votchallenge.net/vot2021/)
  - [Toolbox](#toolbox)
  - [Course](#course)

<!-- /TOC -->
<!---
  <a id="markdown-review-papers" name="review-papers"></a>
  ## Review papers

  Multiple Object Tracking: A Literature Review [[paper](https://arxiv.org/pdf/1409.7618.pdf)]

  Deep Learning in Video Multi-Object Tracking: A Survey [[paper](https://arxiv.org/pdf/1907.12740.pdf)]

  Tracking the Trackers: An Analysis of the State of the Art in Multiple Object Tracking [[paper](https://arxiv.org/pdf/1704.02781.pdf)]

  Machine Learning Methods for Data Association in Multi-Object Tracking [[paper](https://arxiv.org/pdf/1802.06897v2)]

  MOTChallenge: A Benchmark for Single-camera Multiple Target Tracking [[paper](https://arxiv.org/pdf/2010.07548.pdf)]  **new paper for new MOT researcher**

  <a id="markdown-algorithm-papers" name="algorithm-papers"></a>
  ## Algorithm papers

  <a id="markdown-2021" name="2021"></a>
  ### **2021**

  **ByteTrack:** ByteTrack: Multi-Object Tracking by Associating Every Detection Box [[code](https://github.com/ifzhang/ByteTrack)] [[paper](https://arxiv.org/pdf/2110.06864.pdf)]  **new SOTA**

  **PermaTrack**: Learning to Track with Object Permanence [[code](https://github.com/TRI-ML/permatrack)] [[paper](https://arxiv.org/pdf/2103.14258.pdf)]  **ICCV2021**

  **SOTMOT**: Improving Multiple Object Tracking with Single Object Tracking [code] [[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Zheng_Improving_Multiple_Object_Tracking_With_Single_Object_Tracking_CVPR_2021_paper.pdf)] **CVPR2021**

  **LPC_MOT**: Learning a Proposal Classifier for Multiple Object Tracking [[code](https://github.com/daip13/LPC_MOT)] [[paper](https://arxiv.org/pdf/2103.07889.pdf)] **CVPR2021**

  **MTP**: Discriminative Appearance Modeling with Multi-track Pooling for Real-time Multi-object Tracking [[code](https://github.com/chkim403/blstm-mtp)] [[paper](https://arxiv.org/pdf/2101.12159.pdf)] **CVPR2021**

  **TADAM**: Online Multiple Object Tracking with Cross-Task Synergy [[code](https://github.com/songguocode/TADAM)] [[paper](https://arxiv.org/pdf/2104.00380.pdf)] **CVPR2021**

  **RelationTrack**: RelationTrack: Relation-aware Multiple Object Tracking with Decoupled Representation [[code]] [[paper](https://arxiv.org/pdf/2105.04322.pdf)]

  **MOTR**: MOTR: End-to-End Multiple-Object Tracking with TRansformer [[code](https://github.com/megvii-model/MOTR)]  [[paper](https://arxiv.org/pdf/2105.03247.pdf)]

  **CSTrackV2/RCNet**: One More Check: Making "Fake Background" Be Tracked Again [[code](https://github.com/JudasDie/SOTS)] [[paper](https://arxiv.org/pdf/2104.09441.pdf)]

  **QDTrack**: Quasi-Dense Similarity Learning for Multiple Object Tracking [[code](https://github.com/SysCV/qdtrack)] [[paper](https://arxiv.org/pdf/2006.06664.pdf)] **CVPR2021**

  **SiamMOT**: SiamMOT: Siamese Multi-Object Tracking [[code](https://github.com/amazon-research/siam-mot)] [[paper](https://arxiv.org/pdf/2105.11595.pdf)] **CVPR2021**

  **GMTracker**: Learnable Graph Matching: Incorporating Graph Partitioning with Deep Feature Learning for Multiple Object Tracking [[code](https://github.com/jiaweihe1996/GMTracker)] [[paper](https://arxiv.org/pdf/2103.16178.pdf)] **CVPR2021**

  **ArTIST**: Probabilistic Tracklet Scoring and Inpainting for Multiple Object Tracking [[paper](https://arxiv.org/pdf/2012.02337.pdf)] **CVPR2021**

  **CorrTracker/TLR**: Multiple Object Tracking with Correlation Learning [code] [[paper](https://arxiv.org/pdf/2104.03541.pdf)] **CVPR2021**

  **TransMOT**：Spatial-Temporal Graph Transformer for Multiple Object Tracking [code] [[paper](https://arxiv.org/pdf/2104.00194v2.pdf)]

  **TransCenter**: TransCenter: Transformers with Dense Queries for Multiple-Object Tracking [code] [[paper](https://arxiv.org/pdf/2103.15145.pdf)]

  **GCNet**: Global Correlation Network: End-to-End Joint Multi-Object Detection and Tracking [code] [[paper](https://arxiv.org/pdf/2103.12511.pdf)]

  **TraDes**: Track to Detect and Segment: An Online Multi-Object Tracker [[code](https://github.com/JialianW/TraDeS)]  [[paper](https://arxiv.org/pdf/2103.08808.pdf)] **CVPR2021**

  **DEFT**: DEFT: Detection Embeddings for Tracking [[code](https://github.com/MedChaabane/DEFT)] [[paper](https://arxiv.org/pdf/2102.02267.pdf)]

  **TrackMPNN**: TrackMPNN: A Message Passing Graph Neural Architecture for Multi-Object Tracking [[code](https://github.com/arangesh/TrackMPNN)] [[paper](https://arxiv.org/pdf/2101.04206.pdf)]

  **TrackFormer**: TrackFormer: Multi-Object Tracking with Transformers [[code]] [[paper](https://arxiv.org/pdf/2101.02702.pdf)]


  <a id="markdown-2020" name="2020"></a>
  ### **2020**

  **TransTrack**: TransTrack: Multiple-Object Tracking with Transformer [[code](https://github.com/PeizeSun/TransTrack)] [[paper](https://arxiv.org/pdf/2012.15460.pdf)]

  **TPAGT**: Tracklets Predicting Based Adaptive Graph Tracking [[paper](https://arxiv.org/pdf/2010.09015v3.pdf)] **original FGAGT**

  **MLT**: Multiplex Labeling Graph for Near-Online Tracking in Crowded Scenes [[paper](https://ieeexplore.ieee.org/document/9098857)]

  **GSDT**: Joint Object Detection and Multi-Object Tracking with Graph Neural Networks [[paper](http://arxiv.org/pdf/2006.13164)]

  **SMOT**: SMOT: Single-Shot Multi Object Tracking [[paper](http://arxiv.org/pdf/2010.16031)]

  **CSTrack**: Rethinking the competition between detection and ReID in Multi-Object Tracking [[code](https://github.com/JudasDie/SOTS)][[paper](http://arxiv.org/pdf/2010.12138)]

  **MAT**: MAT: Motion-Aware Multi-Object Tracking [[paper](https://arxiv.org/ftp/arxiv/papers/2009/2009.04794.pdf)]

  **UnsupTrack**: Simple Unsupervised Multi-Object Tracking [[paper](https://arxiv.org/pdf/2006.02609.pdf)]

  **FairMOT**: FairMOT: On the Fairness of Detection and Re-Identification in Multiple Object Tracking [[code](https://github.com/ifzhang/FairMOT)][[paper](https://arxiv.org/pdf/2004.01888v5.pdf)] **a new version of FairMOT, compared with new method like CTracker**

  **DMM-Net**: Simultaneous Detection and Tracking with Motion Modelling for Multiple Object Tracking [[code](https://github.com/shijieS/DMMN)][[paper](https://arxiv.org/abs/2008.08826)]

  **SoDA**: SoDA: Multi-Object Tracking with Soft Data Association [[code]][[paper](https://arxiv.org/abs/2008.07725)]

  **CTracker**: Chained-Tracker: Chaining Paired Attentive Regression Results for End-to-End Joint Multiple-Object Detection and Tracking [[code](https://github.com/pjl1995/CTracker)][[paper](http://arxiv.org/abs/2007.14557)]

  **MPNTracker**: Learning a Neural Solver for Multiple Object Tracking [[code]](https://github.com/dvl-tum/mot_neural_solver)[[paper]](https://arxiv.org/pdf/1912.07515.pdf)

  **UMA**: A Unified Object Motion and Affinity Model for Online Multi-Object Tracking [[code]](https://github.com/yinjunbo/UMA-MOT)[[paper]](https://arxiv.org/pdf/2003.11291.pdf)

  **RetinaTrack**: Online Single Stage Joint Detection and Tracking [[code]][[paper]](https://arxiv.org/pdf/2003.13870.pdf)

  **FairMOT**: A Simple Baseline for Multi-Object Tracking [[code]](https://github.com/ifzhang/FairMOT)[[paper]](https://arxiv.org/pdf/2004.01888.pdf)

  **TubeTK**: TubeTK: Adopting Tubes to Track Multi-Object in a One-Step Training Model [[code](https://github.com/BoPang1996/TubeTK)][[paper](https://arxiv.org/pdf/2006.05683.pdf)]

  **CenterTrack**: Tracking Objects as Points [[code]](https://github.com/xingyizhou/CenterTrack)[[paper]](https://arxiv.org/pdf/2004.01177.pdf)

  **Lif_T**: Lifted Disjoint Paths with Application in Multiple Object Tracking [[code]](https://github.com/AndreaHor/LifT_Solver)[[paper]](https://arxiv.org/pdf/2006.14550.pdf)

  **PointTrack**: Segment as points for efficient online multi-object tracking and segmentation [[code]](https://github.com/detectRecog/PointTrack)[[paper]](https://arxiv.org/pdf/2007.01550.pdf)

  **PointTrack++**: PointTrack++ for Effective Online Multi-Object Tracking and Segmentation [[code]](https://github.com/detectRecog/PointTrack)[[paper]](https://arxiv.org/pdf/2007.01549.pdf)

  **FFT**: Multiple Object Tracking by Flowing and Fusing [[paper]](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2001.11180)

  **MIFT**: Refinements in Motion and Appearance for Online Multi-Object Tracking [[code]](https://github.com/nightmaredimple/libmot)[[paper]](https://arxiv.org/pdf/2003.07177.pdf)

  **EDA_GNN**: Graph Neural Based End-to-end Data Association Framework for Online Multiple-Object Tracking [[code]](https://github.com/peizhaoli05/EDA_GNN)[[paper]](https://arxiv.org/pdf/1907.05315.pdf)

  **GNMOT**: Graph Networks for Multiple Object Tracking [[code]](https://github.com/yinizhizhu/GNMOT)[[paper]](https://openaccess.thecvf.com/content_WACV_2020/html/Li_Graph_Networks_for_Multiple_Object_Tracking_WACV_2020_paper.html)

  <a id="markdown-2019" name="2019"></a>
  ### **2019**

  **Tracktor/Tracktor++**: Tracking without bells and whistles [[code]](https://github.com/phil-bergmann/tracking_wo_bnw)[[paper]](https://arxiv.org/pdf/1903.05625.pdf)

  **DeepMOT**: How To Train Your Deep Multi-Object Tracker [[code]](https://github.com/yihongXU/deepMOT)[[paper]](https://arxiv.org/pdf/1906.06618.pdf)

  **JDE**: Towards Real-Time Multi-Object Tracking [[code]](https://github.com/Zhongdao/Towards-Realtime-MOT)[[paper]](https://arxiv.org/pdf/1909.12605v1.pdf)

  **MOTS**: MOTS: Multi-Object Tracking and Segmentation[[paper]](https://arxiv.org/pdf/1902.03604.pdf)

  **FANTrack**: FANTrack: 3D Multi-Object Tracking with Feature Association Network [[code]](https://git.uwaterloo.ca/wise-lab/fantrack)[[paper]](https://arxiv.org/pdf/1905.02843.pdf)

  **FAMNet**: Joint Learning of Feature, Affinity and Multi-dimensional Assignment for Online Multiple Object Tracking[[paper]](https://arxiv.org/pdf/1904.04989.pdf)

  <a id="markdown-2018" name="2018"></a>
  ### **2018**

  **DeepCC**: Features for Multi-Target Multi-Camera Tracking and Re-Identification [[paper]](https://openaccess.thecvf.com/content_cvpr_2018/papers/Ristani_Features_for_Multi-Target_CVPR_2018_paper.pdf)

  **SADF**: Online Multi-Object Tracking with Historical Appearance Matching and Scene Adaptive Detection Filtering [[paper]](https://arxiv.org/pdf/1805.10916.pdf)

  **DAN**: Deep Affinity Network for Multiple Object Tracking [[code]](https://link.zhihu.com/?target=https%3A//github.com/shijieS/SST.git)[[paper]](https://link.zhihu.com/?target=http%3A//arxiv.org/abs/1810.11780)

  **DMAN**: Online Multi-Object Tracking with Dual Matching Attention Networks [[code]](https://github.com/jizhu1023/DMAN_MOT)[[paper]](https://openaccess.thecvf.com/content_ECCV_2018/papers/Ji_Zhu_Online_Multi-Object_Tracking_ECCV_2018_paper.pdf)

  **MOTBeyondPixels**: Beyond Pixels: Leveraging Geometry and Shape Cues for Online Multi-Object Tracking [[code]](https://github.com/JunaidCS032/MOTBeyondPixels)[[paper]](http://arxiv.org/abs/1802.09298)

  **MOTDT**: Real-time Multiple People Tracking with Deeply Learned Candidate Selection and Person Re-Identification [[code]](https://github.com/longcw/MOTDT)[[paper]](https://arxiv.org/abs/1809.04427)

  **DetTA**: Detection-Tracking for Efficient Person Analysis: The DetTA Pipeline [[code]](https://github.com/sbreuers/detta)[[paper]](https://arxiv.org/abs/1804.10134)

  **V-IOU**: Extending IOU Based Multi-Object Tracking by Visual Information [[code]](https://github.com/bochinski/iou-tracker/)[[paper]](http://elvera.nue.tu-berlin.de/files/1547Bochinski2018.pdf)


  <a id="markdown-2017" name="2017"></a>
  ### **2017**

  **DeepSORT**: Simple Online and Realtime Tracking with a Deep Association Metric [[code]](https://github.com/nwojke/deep_sort)[[paper]](https://arxiv.org/pdf/1703.07402.pdf)

  **NMGC-MOT**: Non-Markovian Globally Consistent Multi-Object Tracking [[code]](https://github.com/maksay/ptrack_cpp)[[paper]](https://openaccess.thecvf.com/content_ICCV_2017/papers/Maksai_Non-Markovian_Globally_Consistent_ICCV_2017_paper.pdf)

  **IOUTracker**: High-Speed tracking-by-detection without using image information [[code]](https://github.com/bochinski/iou-tracker/)[[paper]](http://elvera.nue.tu-berlin.de/typo3/files/1517Bochinski2017.pdf)

  **RNN_LSTM**: Online Multi-Target Tracking Using Recurrent Neural Networks [[code]](https://bitbucket.org/amilan/rnntracking)[[paper]](https://arxiv.org/abs/1604.03635)

  **D2T**: Detect to Track and Track to Detect [[code]](https://github.com/feichtenhofer/Detect-Track)[[paper]](https://openaccess.thecvf.com/content_ICCV_2017/papers/Feichtenhofer_Detect_to_Track_ICCV_2017_paper.pdf)

  **RCMSS**: Online multi-object tracking via robust collaborative model and sample selection [[paper]](https://faculty.ucmerced.edu/mhyang/papers/cviu16_MOT.pdf)

  **towards-reid-tracking**: Towards a Principled Integration of Multi-Camera Re-Identification and Tracking through Optimal Bayes Filters [[code]](https://github.com/VisualComputingInstitute/towards-reid-tracking)[[paper]](https://arxiv.org/pdf/1705.04608.pdf)

  **CIWT**: Combined image-and world-space tracking in traffic scenes [[code]](https://github.com/aljosaosep/ciwt)[[paper]](https://arxiv.org/pdf/1809.07357.pdf)


  <a id="markdown-2016" name="2016"></a>
  ### **2016**

  **SORT**: Simple online and realtime tracking [[code]](https://link.zhihu.com/?target=https%3A//github.com/abewley/sort)[[paper]](https://arxiv.org/pdf/1602.00763.pdf)

  **POI**: POI: Multiple Object Tracking with High Performance Detection and Appearance Feature [[code](https://arxiv.org/pdf/1610.06136)]


  <a id="markdown-datasets" name="datasets"></a>
  ## Datasets


  <a id="markdown-surveillance-scenarios" name="surveillance-scenarios"></a>
  ### Surveillance Scenarios

  PETS 2009 Benchmark Data [[url]](http://www.cvg.reading.ac.uk/PETS2009/a.html)<br>
  MOT Challenge [[url]](https://motchallenge.net/)<br>
  UA-DETRAC [[url]](http://detrac-db.rit.albany.edu/download)<br>
  WILDTRACK [[url]](https://www.epfl.ch/labs/cvlab/data/data-wildtrack/)<br>
  NVIDIA AI CITY Challenge [[url]](https://www.aicitychallenge.org/2020-data-and-evaluation/)<br>
  VisDrone [[url]](https://github.com/VisDrone)<br>
  JTA Dataset [[url]](https://github.com/fabbrimatteo/JTA-Dataset)<br>
  Path Track [[url]](https://www.trace.ethz.ch/publications/2017/pathtrack/index.html)<br>
  TAO [[url]](https://github.com/TAO-Dataset/tao)<br>
  GMOT40 [[url]](https://arxiv.org/abs/2011.11858)<br>


  <a id="markdown-driving-scenarios" name="driving-scenarios"></a>
  ### Driving Scenarios

  KITTI-Tracking [[url]](http://www.cvlibs.net/datasets/kitti/eval_tracking.php)<br>
  APOLLOSCAPE [[url]](http://apolloscape.auto/tracking.html)<br>
  APOLLO MOTS [[url]](https://github.com/detectRecog/PointTrack)<br>
  Omni-MOT [[url]](https://pan.baidu.com/s/1ma0rZIW6vfXeq5tdEk6K2w)<br>
  BDD100K [[url]](http://bdd-data.berkeley.edu/)<br>
  Waymo [[url]](https://waymo.com/open/download/#)<br>




  <a id="markdown-metrics" name="metrics"></a>
  ## Metrics

  | metric|formula|
  | :---:|:---:|
  | accuracy| $ Accuracy = {{TP + TN} \over {TP + TN + FP + FN}} $|
  | recall | $ Recall = {TP \over {TP + FN}} = TPR$|
  |precision|$ Precision = {TP \over {TP + FP}} $|
  |MA|$ MA = {FN \over {TP + FN}} $|
  |FA| $ FA = {FP \over {TP + FP}} $|
  | MOTA| $MOTA = 1 - {\sum_t(FN + FP + IDs)\over \sum_t gt}$|
  |MOTP|$ MOTP = {\sum_{t,i}d_t^i \over \sum_tc_t }$|
  |IDP|$ IDP = {IDTP \over {IDTP + IDFP}} $|
  |IDR| $ IDR = {IDTP \over {IDTP + IDFN}} $|
  |IDF1| $ IDF1 = {2 \over {{1 \over IDP} + {1 \over IDR}}} = {2IDTP \over {2IDTP + IDFP + IDFN}} $|

  [Evaluation code](https://github.com/cheind/py-motmetrics)

  <a id="markdown-benchmark-results" name="benchmark-results"></a>
  ## Benchmark Results

  <a id="markdown-mot16" name="mot16"></a>
  ### 

  | Rank |      Model       | MOTA |                                                    Paper                                                    | Year |
  | :--: | :--------------: | :--: | :---------------------------------------------------------------------------------------------------------: | :--: |
  |  1   |                  | 68.7 |                                 A Simple Baseline for Multi-Object Tracking                                 | 2020 |
  |  2   |       JDE        | 64.4 |                                   Towards Real-Time Multi-Object Tracking                                   | 2019 |
  |  3   |      Lif_T       | 61.3 |                     Lifted Disjoint Paths with Application in Multiple Object Tracking                      | 2020 |
  |  4   |     MPNTrack     | 58.6 |                            Learning a Neural Solver for Multiple Object Tracking                            | 2020 |
  |  5   | DeepMOT-Tracktor | 54.8 |                                 How To Train Your Deep Multi-Object Tracker                                 | 2019 |
  |  6   |       TNT        | 49.2 |                      Exploit the Connectivity: Multi-Object Tracking with TrackletNet                       | 2018 |
  |  7   |       GCRA       | 48.2 | Trajectory Factory: Tracklet Cleaving and Re-connection by Deep Siamese Bi-GRU for Multiple Object Tracking | 2018 |
  |  8   |       FWT        | 47.8 |                      Fusion of Head and Full-Body Detectors for Multi-Object Tracking                       | 2017 |
  |  9   |      MOTDT       | 47.6 |   Real-time Multiple People Tracking with Deeply Learned Candidate Selection and Person Re-Identification   | 2018 |
  |  10  |       NOMT       | 46.4 |                   Near-Online Multi-target Tracking with Aggregated Local Flow Descriptor                   | 2015 |
  |  11  |      DMMOT       | 46.1 |                     Online Multi-Object Tracking with Dual Matching Attention Networks                      | 2019 |

  <a id="markdown-mot17" name="mot17"></a>
  ### MOT17

  | Rank |       Model       | MOTA |                                                    Paper                                                     | Year |
  | :--: | :--------------: | :--: | :---------------------------------------------------------------------------------------------------------: | :--: |
  |  1   |     FairMOT      | 67.5 |                                 A Simple Baseline for Multi-Object Tracking                                 | 2020 |
  |  2   |       Lif_T        | 60.5 |                                   Lifted Disjoint Paths with Application in Multiple Object Tracking                                   | 2020 |
  |3|MPNTrack| 58.8 | Learning a Neural Solver for Multiple Object Tracking | 2020|
  |4| DeepMOT | 53.7|How To Train Your Deep Multi-Object Tracker|2019|
  |5| JBNOT|52.6| Multiple People Tracking using Body and Joint Detections|2019|
  |6|TNT|51.9|Exploit the Connectivity: Multi-Object Tracking with TrackletNet|2018|
  |7|	FWT|51.3|Fusion of Head and Full-Body Detectors for Multi-Object Tracking|2017|
  |8|MOTDT17|50.9|Real-time Multiple People Tracking with Deeply Learned Candidate Selection and Person Re-Identification|2018|

  <a id="markdown-mot20" name="mot20"></a>
  ### MOT20


  | Rank |       Model       | MOTA |                                                    Paper                                                     | Year |
  | :--: | :--------------: | :--: | :---------------------------------------------------------------------------------------------------------: | :--: |
  |  1   |     FairMOT      | 61.8 |                                 A Simple Baseline for Multi-Object Tracking                                 | 2020 |
  |2| UnsupTrack| 53.6 |Simple Unsupervised Multi-Object Tracking|2020|

  <a id="markdown-toolbox" name="toolbox"></a>
  ## Toolbox

  **mmtracking**: OpenMMLab Video Perception Toolbox. It supports Single Object Tracking (SOT), Multiple Object Tracking (MOT), Video Object Detection (VID) with a unified framework.

  [Github](https://github.com/open-mmlab/mmtracking)  [DOC](https://mmtracking.readthedocs.io/en/latest/)

  <a id="markdown-course" name="course"></a>
  ## Course

  [link](https://www.youtube.com/watch?v=ay_QLAHcZLY&list=PLadnyz93xCLhSlm2tMYJSKaik39EZV_Uk) is a good course about multiple object tracking. The course is offered as a Massive Open Online Course (MOOC) on edX. 
--->


<!-- /TOC -->
<!---
  <a id="markdown-review-papers" name="review-papers"></a>
  ## Review papers

  Multiple Object Tracking: A Literature Review [[paper](https://arxiv.org/pdf/1409.7618.pdf)]

  Deep Learning in Video Multi-Object Tracking: A Survey [[paper](https://arxiv.org/pdf/1907.12740.pdf)]

  Tracking the Trackers: An Analysis of the State of the Art in Multiple Object Tracking [[paper](https://arxiv.org/pdf/1704.02781.pdf)]

  Machine Learning Methods for Data Association in Multi-Object Tracking [[paper](https://arxiv.org/pdf/1802.06897v2)]

  MOTChallenge: A Benchmark for Single-camera Multiple Target Tracking [[paper](https://arxiv.org/pdf/2010.07548.pdf)]  **new paper for new MOT researcher**

  <a id="markdown-algorithm-papers" name="algorithm-papers"></a>
  ## Algorithm papers

  <a id="markdown-2021" name="2021"></a>
  ### **2021**

  **ByteTrack:** ByteTrack: Multi-Object Tracking by Associating Every Detection Box [[code](https://github.com/ifzhang/ByteTrack)] [[paper](https://arxiv.org/pdf/2110.06864.pdf)]  **new SOTA**

  **PermaTrack**: Learning to Track with Object Permanence [[code](https://github.com/TRI-ML/permatrack)] [[paper](https://arxiv.org/pdf/2103.14258.pdf)]  **ICCV2021**

  **SOTMOT**: Improving Multiple Object Tracking with Single Object Tracking [code] [[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Zheng_Improving_Multiple_Object_Tracking_With_Single_Object_Tracking_CVPR_2021_paper.pdf)] **CVPR2021**

  **LPC_MOT**: Learning a Proposal Classifier for Multiple Object Tracking [[code](https://github.com/daip13/LPC_MOT)] [[paper](https://arxiv.org/pdf/2103.07889.pdf)] **CVPR2021**

  **MTP**: Discriminative Appearance Modeling with Multi-track Pooling for Real-time Multi-object Tracking [[code](https://github.com/chkim403/blstm-mtp)] [[paper](https://arxiv.org/pdf/2101.12159.pdf)] **CVPR2021**

  **TADAM**: Online Multiple Object Tracking with Cross-Task Synergy [[code](https://github.com/songguocode/TADAM)] [[paper](https://arxiv.org/pdf/2104.00380.pdf)] **CVPR2021**

  **RelationTrack**: RelationTrack: Relation-aware Multiple Object Tracking with Decoupled Representation [[code]] [[paper](https://arxiv.org/pdf/2105.04322.pdf)]

  **MOTR**: MOTR: End-to-End Multiple-Object Tracking with TRansformer [[code](https://github.com/megvii-model/MOTR)]  [[paper](https://arxiv.org/pdf/2105.03247.pdf)]

  **CSTrackV2/RCNet**: One More Check: Making "Fake Background" Be Tracked Again [[code](https://github.com/JudasDie/SOTS)] [[paper](https://arxiv.org/pdf/2104.09441.pdf)]

  **QDTrack**: Quasi-Dense Similarity Learning for Multiple Object Tracking [[code](https://github.com/SysCV/qdtrack)] [[paper](https://arxiv.org/pdf/2006.06664.pdf)] **CVPR2021**

  **SiamMOT**: SiamMOT: Siamese Multi-Object Tracking [[code](https://github.com/amazon-research/siam-mot)] [[paper](https://arxiv.org/pdf/2105.11595.pdf)] **CVPR2021**

  **GMTracker**: Learnable Graph Matching: Incorporating Graph Partitioning with Deep Feature Learning for Multiple Object Tracking [[code](https://github.com/jiaweihe1996/GMTracker)] [[paper](https://arxiv.org/pdf/2103.16178.pdf)] **CVPR2021**

  **ArTIST**: Probabilistic Tracklet Scoring and Inpainting for Multiple Object Tracking [[paper](https://arxiv.org/pdf/2012.02337.pdf)] **CVPR2021**

  **CorrTracker/TLR**: Multiple Object Tracking with Correlation Learning [code] [[paper](https://arxiv.org/pdf/2104.03541.pdf)] **CVPR2021**

  **TransMOT**：Spatial-Temporal Graph Transformer for Multiple Object Tracking [code] [[paper](https://arxiv.org/pdf/2104.00194v2.pdf)]

  **TransCenter**: TransCenter: Transformers with Dense Queries for Multiple-Object Tracking [code] [[paper](https://arxiv.org/pdf/2103.15145.pdf)]

  **GCNet**: Global Correlation Network: End-to-End Joint Multi-Object Detection and Tracking [code] [[paper](https://arxiv.org/pdf/2103.12511.pdf)]

  **TraDes**: Track to Detect and Segment: An Online Multi-Object Tracker [[code](https://github.com/JialianW/TraDeS)]  [[paper](https://arxiv.org/pdf/2103.08808.pdf)] **CVPR2021**

  **DEFT**: DEFT: Detection Embeddings for Tracking [[code](https://github.com/MedChaabane/DEFT)] [[paper](https://arxiv.org/pdf/2102.02267.pdf)]

  **TrackMPNN**: TrackMPNN: A Message Passing Graph Neural Architecture for Multi-Object Tracking [[code](https://github.com/arangesh/TrackMPNN)] [[paper](https://arxiv.org/pdf/2101.04206.pdf)]

  **TrackFormer**: TrackFormer: Multi-Object Tracking with Transformers [[code]] [[paper](https://arxiv.org/pdf/2101.02702.pdf)]


  <a id="markdown-2020" name="2020"></a>
  ### **2020**

  **TransTrack**: TransTrack: Multiple-Object Tracking with Transformer [[code](https://github.com/PeizeSun/TransTrack)] [[paper](https://arxiv.org/pdf/2012.15460.pdf)]

  **TPAGT**: Tracklets Predicting Based Adaptive Graph Tracking [[paper](https://arxiv.org/pdf/2010.09015v3.pdf)] **original FGAGT**

  **MLT**: Multiplex Labeling Graph for Near-Online Tracking in Crowded Scenes [[paper](https://ieeexplore.ieee.org/document/9098857)]

  **GSDT**: Joint Object Detection and Multi-Object Tracking with Graph Neural Networks [[paper](http://arxiv.org/pdf/2006.13164)]

  **SMOT**: SMOT: Single-Shot Multi Object Tracking [[paper](http://arxiv.org/pdf/2010.16031)]

  **CSTrack**: Rethinking the competition between detection and ReID in Multi-Object Tracking [[code](https://github.com/JudasDie/SOTS)][[paper](http://arxiv.org/pdf/2010.12138)]

  **MAT**: MAT: Motion-Aware Multi-Object Tracking [[paper](https://arxiv.org/ftp/arxiv/papers/2009/2009.04794.pdf)]

  **UnsupTrack**: Simple Unsupervised Multi-Object Tracking [[paper](https://arxiv.org/pdf/2006.02609.pdf)]

  **FairMOT**: FairMOT: On the Fairness of Detection and Re-Identification in Multiple Object Tracking [[code](https://github.com/ifzhang/FairMOT)][[paper](https://arxiv.org/pdf/2004.01888v5.pdf)] **a new version of FairMOT, compared with new method like CTracker**

  **DMM-Net**: Simultaneous Detection and Tracking with Motion Modelling for Multiple Object Tracking [[code](https://github.com/shijieS/DMMN)][[paper](https://arxiv.org/abs/2008.08826)]

  **SoDA**: SoDA: Multi-Object Tracking with Soft Data Association [[code]][[paper](https://arxiv.org/abs/2008.07725)]

  **CTracker**: Chained-Tracker: Chaining Paired Attentive Regression Results for End-to-End Joint Multiple-Object Detection and Tracking [[code](https://github.com/pjl1995/CTracker)][[paper](http://arxiv.org/abs/2007.14557)]

  **MPNTracker**: Learning a Neural Solver for Multiple Object Tracking [[code]](https://github.com/dvl-tum/mot_neural_solver)[[paper]](https://arxiv.org/pdf/1912.07515.pdf)

  **UMA**: A Unified Object Motion and Affinity Model for Online Multi-Object Tracking [[code]](https://github.com/yinjunbo/UMA-MOT)[[paper]](https://arxiv.org/pdf/2003.11291.pdf)

  **RetinaTrack**: Online Single Stage Joint Detection and Tracking [[code]][[paper]](https://arxiv.org/pdf/2003.13870.pdf)

  **FairMOT**: A Simple Baseline for Multi-Object Tracking [[code]](https://github.com/ifzhang/FairMOT)[[paper]](https://arxiv.org/pdf/2004.01888.pdf)

  **TubeTK**: TubeTK: Adopting Tubes to Track Multi-Object in a One-Step Training Model [[code](https://github.com/BoPang1996/TubeTK)][[paper](https://arxiv.org/pdf/2006.05683.pdf)]

  **CenterTrack**: Tracking Objects as Points [[code]](https://github.com/xingyizhou/CenterTrack)[[paper]](https://arxiv.org/pdf/2004.01177.pdf)

  **Lif_T**: Lifted Disjoint Paths with Application in Multiple Object Tracking [[code]](https://github.com/AndreaHor/LifT_Solver)[[paper]](https://arxiv.org/pdf/2006.14550.pdf)

  **PointTrack**: Segment as points for efficient online multi-object tracking and segmentation [[code]](https://github.com/detectRecog/PointTrack)[[paper]](https://arxiv.org/pdf/2007.01550.pdf)

  **PointTrack++**: PointTrack++ for Effective Online Multi-Object Tracking and Segmentation [[code]](https://github.com/detectRecog/PointTrack)[[paper]](https://arxiv.org/pdf/2007.01549.pdf)

  **FFT**: Multiple Object Tracking by Flowing and Fusing [[paper]](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/2001.11180)

  **MIFT**: Refinements in Motion and Appearance for Online Multi-Object Tracking [[code]](https://github.com/nightmaredimple/libmot)[[paper]](https://arxiv.org/pdf/2003.07177.pdf)

  **EDA_GNN**: Graph Neural Based End-to-end Data Association Framework for Online Multiple-Object Tracking [[code]](https://github.com/peizhaoli05/EDA_GNN)[[paper]](https://arxiv.org/pdf/1907.05315.pdf)

  **GNMOT**: Graph Networks for Multiple Object Tracking [[code]](https://github.com/yinizhizhu/GNMOT)[[paper]](https://openaccess.thecvf.com/content_WACV_2020/html/Li_Graph_Networks_for_Multiple_Object_Tracking_WACV_2020_paper.html)

  <a id="markdown-2019" name="2019"></a>
  ### **2019**

  **Tracktor/Tracktor++**: Tracking without bells and whistles [[code]](https://github.com/phil-bergmann/tracking_wo_bnw)[[paper]](https://arxiv.org/pdf/1903.05625.pdf)

  **DeepMOT**: How To Train Your Deep Multi-Object Tracker [[code]](https://github.com/yihongXU/deepMOT)[[paper]](https://arxiv.org/pdf/1906.06618.pdf)

  **JDE**: Towards Real-Time Multi-Object Tracking [[code]](https://github.com/Zhongdao/Towards-Realtime-MOT)[[paper]](https://arxiv.org/pdf/1909.12605v1.pdf)

  **MOTS**: MOTS: Multi-Object Tracking and Segmentation[[paper]](https://arxiv.org/pdf/1902.03604.pdf)

  **FANTrack**: FANTrack: 3D Multi-Object Tracking with Feature Association Network [[code]](https://git.uwaterloo.ca/wise-lab/fantrack)[[paper]](https://arxiv.org/pdf/1905.02843.pdf)

  **FAMNet**: Joint Learning of Feature, Affinity and Multi-dimensional Assignment for Online Multiple Object Tracking[[paper]](https://arxiv.org/pdf/1904.04989.pdf)

  <a id="markdown-2018" name="2018"></a>
  ### **2018**

  **DeepCC**: Features for Multi-Target Multi-Camera Tracking and Re-Identification [[paper]](https://openaccess.thecvf.com/content_cvpr_2018/papers/Ristani_Features_for_Multi-Target_CVPR_2018_paper.pdf)

  **SADF**: Online Multi-Object Tracking with Historical Appearance Matching and Scene Adaptive Detection Filtering [[paper]](https://arxiv.org/pdf/1805.10916.pdf)

  **DAN**: Deep Affinity Network for Multiple Object Tracking [[code]](https://link.zhihu.com/?target=https%3A//github.com/shijieS/SST.git)[[paper]](https://link.zhihu.com/?target=http%3A//arxiv.org/abs/1810.11780)

  **DMAN**: Online Multi-Object Tracking with Dual Matching Attention Networks [[code]](https://github.com/jizhu1023/DMAN_MOT)[[paper]](https://openaccess.thecvf.com/content_ECCV_2018/papers/Ji_Zhu_Online_Multi-Object_Tracking_ECCV_2018_paper.pdf)

  **MOTBeyondPixels**: Beyond Pixels: Leveraging Geometry and Shape Cues for Online Multi-Object Tracking [[code]](https://github.com/JunaidCS032/MOTBeyondPixels)[[paper]](http://arxiv.org/abs/1802.09298)

  **MOTDT**: Real-time Multiple People Tracking with Deeply Learned Candidate Selection and Person Re-Identification [[code]](https://github.com/longcw/MOTDT)[[paper]](https://arxiv.org/abs/1809.04427)

  **DetTA**: Detection-Tracking for Efficient Person Analysis: The DetTA Pipeline [[code]](https://github.com/sbreuers/detta)[[paper]](https://arxiv.org/abs/1804.10134)

  **V-IOU**: Extending IOU Based Multi-Object Tracking by Visual Information [[code]](https://github.com/bochinski/iou-tracker/)[[paper]](http://elvera.nue.tu-berlin.de/files/1547Bochinski2018.pdf)


  <a id="markdown-2017" name="2017"></a>
  ### **2017**

  **DeepSORT**: Simple Online and Realtime Tracking with a Deep Association Metric [[code]](https://github.com/nwojke/deep_sort)[[paper]](https://arxiv.org/pdf/1703.07402.pdf)

  **NMGC-MOT**: Non-Markovian Globally Consistent Multi-Object Tracking [[code]](https://github.com/maksay/ptrack_cpp)[[paper]](https://openaccess.thecvf.com/content_ICCV_2017/papers/Maksai_Non-Markovian_Globally_Consistent_ICCV_2017_paper.pdf)

  **IOUTracker**: High-Speed tracking-by-detection without using image information [[code]](https://github.com/bochinski/iou-tracker/)[[paper]](http://elvera.nue.tu-berlin.de/typo3/files/1517Bochinski2017.pdf)

  **RNN_LSTM**: Online Multi-Target Tracking Using Recurrent Neural Networks [[code]](https://bitbucket.org/amilan/rnntracking)[[paper]](https://arxiv.org/abs/1604.03635)

  **D2T**: Detect to Track and Track to Detect [[code]](https://github.com/feichtenhofer/Detect-Track)[[paper]](https://openaccess.thecvf.com/content_ICCV_2017/papers/Feichtenhofer_Detect_to_Track_ICCV_2017_paper.pdf)

  **RCMSS**: Online multi-object tracking via robust collaborative model and sample selection [[paper]](https://faculty.ucmerced.edu/mhyang/papers/cviu16_MOT.pdf)

  **towards-reid-tracking**: Towards a Principled Integration of Multi-Camera Re-Identification and Tracking through Optimal Bayes Filters [[code]](https://github.com/VisualComputingInstitute/towards-reid-tracking)[[paper]](https://arxiv.org/pdf/1705.04608.pdf)

  **CIWT**: Combined image-and world-space tracking in traffic scenes [[code]](https://github.com/aljosaosep/ciwt)[[paper]](https://arxiv.org/pdf/1809.07357.pdf)


  <a id="markdown-2016" name="2016"></a>
  ### **2016**

  **SORT**: Simple online and realtime tracking [[code]](https://link.zhihu.com/?target=https%3A//github.com/abewley/sort)[[paper]](https://arxiv.org/pdf/1602.00763.pdf)

  **POI**: POI: Multiple Object Tracking with High Performance Detection and Appearance Feature [[code](https://arxiv.org/pdf/1610.06136)]


  <a id="markdown-datasets" name="datasets"></a>
  ## Datasets


  <a id="markdown-surveillance-scenarios" name="surveillance-scenarios"></a>
  ### Surveillance Scenarios

  PETS 2009 Benchmark Data [[url]](http://www.cvg.reading.ac.uk/PETS2009/a.html)<br>
  MOT Challenge [[url]](https://motchallenge.net/)<br>
  UA-DETRAC [[url]](http://detrac-db.rit.albany.edu/download)<br>
  WILDTRACK [[url]](https://www.epfl.ch/labs/cvlab/data/data-wildtrack/)<br>
  NVIDIA AI CITY Challenge [[url]](https://www.aicitychallenge.org/2020-data-and-evaluation/)<br>
  VisDrone [[url]](https://github.com/VisDrone)<br>
  JTA Dataset [[url]](https://github.com/fabbrimatteo/JTA-Dataset)<br>
  Path Track [[url]](https://www.trace.ethz.ch/publications/2017/pathtrack/index.html)<br>
  TAO [[url]](https://github.com/TAO-Dataset/tao)<br>
  GMOT40 [[url]](https://arxiv.org/abs/2011.11858)<br>


  <a id="markdown-driving-scenarios" name="driving-scenarios"></a>
  ### Driving Scenarios

  KITTI-Tracking [[url]](http://www.cvlibs.net/datasets/kitti/eval_tracking.php)<br>
  APOLLOSCAPE [[url]](http://apolloscape.auto/tracking.html)<br>
  APOLLO MOTS [[url]](https://github.com/detectRecog/PointTrack)<br>
  Omni-MOT [[url]](https://pan.baidu.com/s/1ma0rZIW6vfXeq5tdEk6K2w)<br>
  BDD100K [[url]](http://bdd-data.berkeley.edu/)<br>
  Waymo [[url]](https://waymo.com/open/download/#)<br>




  <a id="markdown-metrics" name="metrics"></a>
  ## Metrics

  | metric|formula|
  | :---:|:---:|
  | accuracy| $ Accuracy = {{TP + TN} \over {TP + TN + FP + FN}} $|
  | recall | $ Recall = {TP \over {TP + FN}} = TPR$|
  |precision|$ Precision = {TP \over {TP + FP}} $|
  |MA|$ MA = {FN \over {TP + FN}} $|
  |FA| $ FA = {FP \over {TP + FP}} $|
  | MOTA| $MOTA = 1 - {\sum_t(FN + FP + IDs)\over \sum_t gt}$|
  |MOTP|$ MOTP = {\sum_{t,i}d_t^i \over \sum_tc_t }$|
  |IDP|$ IDP = {IDTP \over {IDTP + IDFP}} $|
  |IDR| $ IDR = {IDTP \over {IDTP + IDFN}} $|
  |IDF1| $ IDF1 = {2 \over {{1 \over IDP} + {1 \over IDR}}} = {2IDTP \over {2IDTP + IDFP + IDFN}} $|

  [Evaluation code](https://github.com/cheind/py-motmetrics)

  <a id="markdown-benchmark-results" name="benchmark-results"></a>
  ## Benchmark Results

  <a id="markdown-mot16" name="mot16"></a>
  ### 

  | Rank |      Model       | MOTA |                                                    Paper                                                    | Year |
  | :--: | :--------------: | :--: | :---------------------------------------------------------------------------------------------------------: | :--: |
  |  1   |                  | 68.7 |                                 A Simple Baseline for Multi-Object Tracking                                 | 2020 |
  |  2   |       JDE        | 64.4 |                                   Towards Real-Time Multi-Object Tracking                                   | 2019 |
  |  3   |      Lif_T       | 61.3 |                     Lifted Disjoint Paths with Application in Multiple Object Tracking                      | 2020 |
  |  4   |     MPNTrack     | 58.6 |                            Learning a Neural Solver for Multiple Object Tracking                            | 2020 |
  |  5   | DeepMOT-Tracktor | 54.8 |                                 How To Train Your Deep Multi-Object Tracker                                 | 2019 |
  |  6   |       TNT        | 49.2 |                      Exploit the Connectivity: Multi-Object Tracking with TrackletNet                       | 2018 |
  |  7   |       GCRA       | 48.2 | Trajectory Factory: Tracklet Cleaving and Re-connection by Deep Siamese Bi-GRU for Multiple Object Tracking | 2018 |
  |  8   |       FWT        | 47.8 |                      Fusion of Head and Full-Body Detectors for Multi-Object Tracking                       | 2017 |
  |  9   |      MOTDT       | 47.6 |   Real-time Multiple People Tracking with Deeply Learned Candidate Selection and Person Re-Identification   | 2018 |
  |  10  |       NOMT       | 46.4 |                   Near-Online Multi-target Tracking with Aggregated Local Flow Descriptor                   | 2015 |
  |  11  |      DMMOT       | 46.1 |                     Online Multi-Object Tracking with Dual Matching Attention Networks                      | 2019 |

  <a id="markdown-mot17" name="mot17"></a>
  ### MOT17

  | Rank |       Model       | MOTA |                                                    Paper                                                     | Year |
  | :--: | :--------------: | :--: | :---------------------------------------------------------------------------------------------------------: | :--: |
  |  1   |     FairMOT      | 67.5 |                                 A Simple Baseline for Multi-Object Tracking                                 | 2020 |
  |  2   |       Lif_T        | 60.5 |                                   Lifted Disjoint Paths with Application in Multiple Object Tracking                                   | 2020 |
  |3|MPNTrack| 58.8 | Learning a Neural Solver for Multiple Object Tracking | 2020|
  |4| DeepMOT | 53.7|How To Train Your Deep Multi-Object Tracker|2019|
  |5| JBNOT|52.6| Multiple People Tracking using Body and Joint Detections|2019|
  |6|TNT|51.9|Exploit the Connectivity: Multi-Object Tracking with TrackletNet|2018|
  |7|	FWT|51.3|Fusion of Head and Full-Body Detectors for Multi-Object Tracking|2017|
  |8|MOTDT17|50.9|Real-time Multiple People Tracking with Deeply Learned Candidate Selection and Person Re-Identification|2018|

  <a id="markdown-mot20" name="mot20"></a>
  ### MOT20


  | Rank |       Model       | MOTA |                                                    Paper                                                     | Year |
  | :--: | :--------------: | :--: | :---------------------------------------------------------------------------------------------------------: | :--: |
  |  1   |     FairMOT      | 61.8 |                                 A Simple Baseline for Multi-Object Tracking                                 | 2020 |
  |2| UnsupTrack| 53.6 |Simple Unsupervised Multi-Object Tracking|2020|

  <a id="markdown-toolbox" name="toolbox"></a>
  ## Toolbox

  **mmtracking**: OpenMMLab Video Perception Toolbox. It supports Single Object Tracking (SOT), Multiple Object Tracking (MOT), Video Object Detection (VID) with a unified framework.

  [Github](https://github.com/open-mmlab/mmtracking)  [DOC](https://mmtracking.readthedocs.io/en/latest/)

  <a id="markdown-course" name="course"></a>
  ## Course

  [link](https://www.youtube.com/watch?v=ay_QLAHcZLY&list=PLadnyz93xCLhSlm2tMYJSKaik39EZV_Uk) is a good course about multiple object tracking. The course is offered as a Massive Open Online Course (MOOC) on edX. 
--->

