# Colorful Bias

## Table of Contents
- [Introduction](#introduction)
- [Literature Review](#literature-review)
  - [Race/Ethnicity-Annotated Face Datasets](#raceethnicity-annotated-face-datasets)
  - [Image Colorization Research](#image-colorization-research)
  - [Bias Metrics](#bias-metrics)
- [Setup and Reproduction](#setup-and-reproduction)
- [Methods](#methods)
  - [Choosing the Dataset](#choosing-the-dataset)
  - [Sampling Data](#sampling-data)
  - [Downloading Models](#downloading-models)
  - [Computing Metrics](#computing-metrics)
  - [Analyzing Colorization](#analyzing-colorization)
- [Limitations](#limitations)
- [Conclusion](#conclusion)

## Introduction

In 2015, Google Photos faced widespread backlash after its [algorithms mislabeled](https://www.nytimes.com/2023/05/22/technology/ai-photo-labels-google-apple.html) Black people as gorillas (Grant & Hill, 2023). Three years later, the MIT Media Lab found that facial recognition systems had [error rates](https://www.media.mit.edu/articles/study-finds-gender-and-skin-type-bias-in-commercial-artificial-intelligence-systems/) as high as 34% for darker-skinned women compared to less than 1% for lighter-skinned men (Buolamwini, 2018). From image classification to facial recognition, computer vision is infamously flawed. In this research project, I investigated how these issues of fairness manifest in the age of generative AI. In particular, I explored the robustness of generative algorithms for image colorization with respect to skin tone bias. To accomplish this, I conducted a survey of race/ethnicity-annotated face datasets, compiled seminal algorithms for image colorization over the years, researched various formulations of bias metrics, and set up a code framework with statistical tests to rigorously compare the performance of coloring procedures. Through the above work, I sought to shed light on the trend in “colorful” bias, or bias in algorithmic colorization of images containing human skin tones, as seen through algorithms over time.

## Literature Review

### Race/Ethnicity-Annotated Face Datasets

Many race/ethnicity-annotated face datasets have emerged over the years. Some have faced criticism for how their data were provisioned, an issue that has afflicted computer vision and AI more broadly. One such example is MORPH-II, which, as explained in [MORPH-II: Inconsistencies and Cleaning Whitepaper](https://libres.uncg.edu/ir/uncw/f/wangy2017-1.pdf) (Wang et al., 2018), drew from "55,134 mugshots taken between 2003 and late 2007." Beyond collection, datasets also use different localized conceptions of race/ethnicity, a potentially problematic inconsistency highlighted in [Racial Bias within Face Recognition: A Survey](https://dl.acm.org/doi/pdf/10.1145/3705295) (Yucer et al., 2024) and [One Label, One Billion Faces: Usage and Consistency of Racial Categories in Computer Vision](https://arxiv.org/abs/2102.02320) (Khan & Fu, 2021). Still, even if identities could be balanced in a standardized way, [What Should Be Balanced in a "Balanced" Face Recognition Dataset?](https://papers.bmvc2023.org/0235.pdf) (Wu & Bowyer, 2023) notes that this does not ensure balance in "other factors known to impact accuracy, such as head pose, brightness, and image quality." With this context in mind, we provide an overview of a few race/ethnicity-annotated face datasets.

| Name | Year | Active | Count | Standardization | Races/Ethnicities |
| - | - | - | - | - | - |
| [MORPH-II](https://libres.uncg.edu/ir/uncw/f/wangy2017-1.pdf) | 2006 | No | 55134 | No | Asian, Black, Hispanic, White |
| [Face Place](https://sites.google.com/andrew.cmu.edu/tarrlab/stimuli#h.u2lsuc5ur5gt) | 2008 | Yes | 235 | Yes | Asian, Black, Caucasian, Hispanic, Multiracial |
| [Todorov 13125](https://tlab.uchicago.edu/databases/) | 2013 | Yes | 13125 | Yes | Asian, Black, White |
| [CFD](https://www.chicagofaces.org/) | 2015 | Yes | 827+ | Yes | Asian, Black, Latino, White |
| [RFW](http://www.whdeng.cn/RFW/index.html) | 2018 | Yes | 40607 | Yes | African, Asian, Caucasian, Indian |
| [UTKFace](https://github.com/aicip/UTKFace) | 2019  | No | 20000+ | No | Asian, Black, Indian, White |
| [DemogPairs](https://ihupont.github.io/publications/2019-05-16-demogpairs) | 2019 | Yes | 10800 | Yes | Asian, Black, White |
| [DiveFace](https://github.com/BiDAlab/DiveFace) | 2019 | Yes | 150000+ | Yes | Caucasian, East Asian, Sub-Saharan and South Indian |
| [BFW](https://github.com/visionjo/facerec-bias-bfw) | 2020 | No | 20000+ | Yes | Asian, Black, Indian, White |
| [VMER](https://link.springer.com/article/10.1007/s00138-020-01123-z) | 2020 | No | 3000000+ | Yes | African American, Asian Indian, Caucasian Latin, East Asian |
| [FDEA](https://github.com/GZHU-DVL/FDEA) | 2021  | No | 157801 | Yes | African, Asian, Caucasian |
| [FairFace](https://github.com/dchen236/FairFace) | 2021 | Yes | 108501 | No | Asian, Black, Indian, White |
| [FaceARG](https://www.cs.ubbcluj.ro/~dadi/FaceARG-database.html) | 2021 | Yes | 175000+ | Yes | African-American, Asian, Caucasian, Indian |
| [BUPT-BalancedFace](http://www.whdeng.cn/RFW/Trainingdataste.html) | 2022 | Yes | 1300000+ | Yes | African, Asian, Caucasian, Indian |

### Image Colorization Research

Strategies for image colorization have evolved over the years and feature a diversity of AI frameworks as well as user inputs. Some examples of unsupervised methods include focus on random fields (Deshpande et al., 2015; Messaoud et al., 2018), stochastic sampling (Royer et al., 2017), deep neural networks (Cheng et al., 2016; Iizuka et al., 2016; Larsson et al., 2016; Lempitsky et al., 2018; Yoo et al., 2019), encoders and decoders (Deshpande et al., 2017; Kang et al., 2022), convolutional neural networks (Zhang et al., 2016; Zhang et al., 2017; Baldassarre et al., 2017; Zhao et al., 2019), generative adversarial networks (Cao et al., 2017; Vitoria et al., 2020; Wu et al., 2021; Kim et al., 2022), instance-aware coloring (Su et al., 2020; Jin et al., 2021; Cong et al., 2024), transformers (Kumar et al., 2021; Ji et al., 2022; Huang et al., 2022), and transfer learning (Lee et el., 2022). Likewise, supervised methods leverage sample scribbles and strokes (Levin et al., 2004; Yatziv & Shapiro, 2006; Pang et al., 2013; Sangkloy et al., 2017; Zhang et al., 2018; Sun et al., 2019; Zhang et al., 2021; Dou et al., 2022), reference images or patches (Reinhard et al., 2001; Welsh et al., 2002; Irony et al., 2005; Liu et al., 2008; Gupta et al., 2012; Li et al., 2014; He et al., 2018; Xian et al., 2018; Fang et al., 2019; Li et al., 2019; Lee et al., 2020; Xu et al., 2020; Lu et al., 2020; Kim et al., 2021; Li et al., 2021; Yin et al., 2021; Bai et al., 2022; Wang et al., 2022; Zou et al., 2022), target color palettes and pixels (Chang et al., 2015; Frans, 2017; Bahng et al., 2018; Yun et al., 2023), text descriptions (Chen et al., 2018; Manjunatha et al., 2018; Zabari et al., 2023; Chang et al., 2023; Zhang et al., 2023; Yan et al., 2023), and multimodal combinations of these (Lei & Chen, 2019; Liu et al., 2023; Liang et al., 2024; Bozic et al., 2024). Here, we provide an in-depth overview of research papers on image colorization.

| Title | Year | Author(s) | Supervision | Implementation |
| - | - | - | - | - |
| [Color Transfer between Images](https://www.researchgate.net/publication/220518215_Color_Transfer_between_Images) | 2001 | Reinhard et al. | Yes | https://github.com/chia56028/Color-Transfer-between-Images |
| [Transferring Color to Greyscale Images](https://www.researchgate.net/publication/220183710_Transferring_Color_to_Greyscale_Images) | 2002 | Welsh et al. | Yes | https://github.com/h-wang94/ImageColorization |
| [Colorization Using Optimization](https://dl.acm.org/doi/10.1145/1015706.1015780) | 2004 | Levin et al. | Yes | https://github.com/soumik12345/colorization-using-optimization |
| [Colorization by Example](https://dl.acm.org/doi/10.5555/2383654.2383683) | 2005 | Irony et al. | Yes | None |
| [Fast Image and Video Colorization Using Chrominance Blending](https://www.researchgate.net/publication/7109910_Fast_image_and_video_colorization_using_chrominance_blending) | 2006 | Yatziv & Sapiro | Yes | None |
| [Intrinsic Colorization](https://www.semanticscholar.org/paper/Intrinsic-colorization-Liu-Wan/448adcd417c44555cb136be8101ee86ac521fb9f) | 2008 | Liu et al. | Yes | None |
| [Image Colorization Using Similar Images](https://dl.acm.org/doi/10.1145/2393347.2393402) | 2012 | Gupta et al. | Yes | None |
| [Image Colorization Using Sparse Representation](https://www.researchgate.net/publication/261282221_Image_colorization_using_sparse_representation) | 2013 | Pang et al. | Yes | None |
| [Example-based Image Colorization using Locality Consistent Sparse Representation](https://users.cs.cf.ac.uk/Paul.Rosin/resources/papers/colourisation-TIP-postprint.pdf) | 2014 | Li et al. | Yes | None |
| [Learning Large-Scale Automatic Image Colorization](https://openaccess.thecvf.com/content_iccv_2015/papers/Deshpande_Learning_Large-Scale_Automatic_ICCV_2015_paper.pdf) | 2015 | Deshpande et al. | No | https://github.com/aditya12agd5/iccv15_lscolorization |
| [Palette-based Photo Recoloring](https://gfx.cs.princeton.edu/pubs/Chang_2015_PPR/chang2015-palette_small.pdf) | 2015 | Chang et al. | Yes | https://github.com/b-z/photo_recoloring |
| [Deep Colorization](https://www.researchgate.net/publication/301818846_Deep_Colorization) | 2016 | Cheng et al. | No | None |
| [Colorful Image Colorization](https://arxiv.org/abs/1603.08511) | 2016 | Zhang et al. | No | https://github.com/richzhang/colorization |
| [Let There Be Color](https://dl.acm.org/doi/10.1145/2897824.2925974) | 2016 | Iizuka et al. | No | https://github.com/satoshiiizuka/siggraph2016_colorization |
| [Learning Representations for Automatic Colorization](https://arxiv.org/abs/1603.06668) | 2016 | Larsson et al. | No | https://github.com/gustavla/autocolorize |
| [Unsupervised Diverse Colorization via Generative Adversarial Networks](https://arxiv.org/abs/1702.06674) | 2017 | Cao et al. | No | https://github.com/ccyyatnet/COLORGAN |
| [Real-Time User-Guided Image Colorization with Learned Deep Priors](https://arxiv.org/abs/1705.02999) | 2017 | Zhang et al. | Yes | https://github.com/junyanz/interactive-deep-colorization | 
| [Probabilistic Image Colorization](https://www.researchgate.net/publication/316875180_Probabilistic_Image_Colorization) | 2017 | Royer et al. | No | https://github.com/ameroyer/PIC |
| [Outline Colorization through Tandem Adversarial Networks](https://arxiv.org/abs/1704.08834) | 2017 | Frans | Yes | None |
| [Learning Diverse Image Colorization](https://openaccess.thecvf.com/content_cvpr_2017/papers/Deshpande_Learning_Diverse_Image_CVPR_2017_paper.pdf) | 2017 | Deshpande et al. | No | https://github.com/aditya12agd5/divcolor |
| [Image Colorization using CNNs and Inception-ResNet-v2](https://arxiv.org/abs/1712.03400) | 2017 | Baldassarre et al. | No | https://github.com/baldassarreFe/deep-koalarization |
| [Controlling Deep Image Synthesis with Sketch and Color](https://openaccess.thecvf.com/content_cvpr_2017/papers/Sangkloy_Scribbler_Controlling_Deep_CVPR_2017_paper.pdf) | 2017 | Sangkloy et al. | Yes | None |
| [Deep Exemplar-Based Colorization](https://www.researchgate.net/publication/326726827_Deep_exemplar-based_colorization) | 2018 | He et al. | Yes | https://github.com/msracver/Deep-Exemplar-based-Colorization | 
| [Deep Image Prior](https://ieeexplore.ieee.org/document/8579082) | 2018 | Lempitsky et al. | No | https://github.com/DmitryUlyanov/deep-image-prior |
| [DeOldify](https://github.com/jantic/DeOldify) | 2018 | Antic | No | https://github.com/jantic/DeOldify |
| [TextureGAN: Controlling Deep Image Synthesis with Texture Patches](https://openaccess.thecvf.com/content_cvpr_2018/papers/Xian_TextureGAN_Controlling_Deep_CVPR_2018_paper.pdf) | 2018 | Xian et al. | Yes | https://github.com/janesjanes/Pytorch-TextureGAN |
| [Two-Stage Sketch Colorization](https://ttwong12.github.io/papers/colorize/colorize.pdf) | 2018 | Zhang et al. | Yes | None |
| [Learning to Color from Language](https://arxiv.org/pdf/1804.06026) | 2018 | Manjunatha et al. | Yes | https://github.com/superhans/colorfromlanguage |
| [Language-Based Image Editing with Recurrent Attentive Models](https://arxiv.org/pdf/1711.06288) | 2018 | Chen et al. | Yes | https://github.com/Jianbo-Lab/LBIE |
| [Structural Consistency and Controllability for Diverse Colorization](https://openaccess.thecvf.com/content_ECCV_2018/papers/Safa_Messaoud_Structural_Consistency_and_ECCV_2018_paper.pdf) | 2018 | Messaoud et al. | No | None |
| [Awesome Image Colorization](https://github.com/MarkMoHR/Awesome-Image-Colorization) | 2018 | Mo et al. | Yes | https://github.com/MarkMoHR/Awesome-Image-Colorization | 
| [Coloring with Words: Guiding Image Colorization Through Text-based Palette Generation](https://arxiv.org/pdf/1804.04128) | 2018 | Bahng et al. | Yes | https://github.com/awesome-davian/Text2Colors/ |
| [Pixelated Semantic Colorization](https://arxiv.org/abs/1901.10889) | 2019 | Zhao et al. | No | None |
| [Fully Automatic Video Colorization with Self-Regularization and Diversity](https://arxiv.org/pdf/1908.01311) | 2019 | Lei & Chen | No | https://github.com/ChenyangLEI/automatic-video-colorization |
| [A Superpixel-Based Variational Model for Image Colorization](https://ieeexplore.ieee.org/abstract/document/8676327) | 2019 | Fang et al. | Yes | None |
| [Adversarial Colorization Of Icons Based On Structure And Color Conditions](https://arxiv.org/pdf/1910.05253) | 2019 | Sun et al. | Yes | https://github.com/jxcodetw/Adversarial-Colorization-Of-Icons-Based-On-Structure-And-Color-Conditions |
| [Automatic Example-Based Image Colorization Using Location-Aware Cross-Scale Matching](https://ieeexplore.ieee.org/abstract/document/8699109) | 2019 | Li et al. | Yes | None |
| [Coloring With Limited Data: Few-Shot Colorization via Memory Augmented Networks](https://openaccess.thecvf.com/content_CVPR_2019/papers/Yoo_Coloring_With_Limited_Data_Few-Shot_Colorization_via_Memory_Augmented_Networks_CVPR_2019_paper.pdf) | 2019 | Yoo et al. | No | https://github.com/dongheehand/MemoPainter-PyTorch |
| [ChromaGAN: Adversarial Picture Colorization with Semantic Class Distribution](https://arxiv.org/abs/1907.09837) | 2020 | Vitoria et al. | No | https://github.com/pvitoria/ChromaGAN |
| [Instance-Aware Image Colorization](https://arxiv.org/abs/2005.10825) | 2020 | Su et al. | No | https://github.com/ericsujw/InstColorization |
| [Reference-Based Sketch Image Colorization using Augmented-Self Reference and Dense Semantic Correspondence](https://openaccess.thecvf.com/content_CVPR_2020/papers/Lee_Reference-Based_Sketch_Image_Colorization_Using_Augmented-Self_Reference_and_Dense_Semantic_CVPR_2020_paper.pdf) | 2020 | Lee et al. | Yes | None |
| [Stylization-Based Architecture for Fast Deep Exemplar Colorization](https://openaccess.thecvf.com/content_CVPR_2020/papers/Xu_Stylization-Based_Architecture_for_Fast_Deep_Exemplar_Colorization_CVPR_2020_paper.pdf) | 2020 | Xu et al. | Yes | https://github.com/xuzhongyou/Colorization |
| [Image Colorization: A Survey and Dataset](https://arxiv.org/abs/2008.10774) | 2020 | Anwar et al. | No | https://github.com/saeed-anwar/ColorSurvey | 
| [Gray2ColorNet: Transfer More Colors from Reference Image](https://dl.acm.org/doi/abs/10.1145/3394171.3413594) | 2020 | Lu et al. | Yes | https://github.com/CV-xueba/Gray2ColorNet | 
| [Colorization Transformer](https://arxiv.org/abs/2102.04432) | 2021 | Kumar et al. | No | https://github.com/google-research/google-research/tree/master/coltran |
| [Colorizing Old Images Learning from Modern Historical Movies](https://arxiv.org/abs/2108.06515) | 2021 | Jin et al. | No | https://github.com/BestiVictory/HistoryNet |
| [Yes, "Attention Is All You Need", for Exemplar based Colorization](https://dl.acm.org/doi/abs/10.1145/3474085.3475385) | 2021 | Yin et al. | Yes | None |
| [User-Guided Line Art Flat Filling with Split Filling Mechanism](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_User-Guided_Line_Art_Flat_Filling_With_Split_Filling_Mechanism_CVPR_2021_paper.pdf) | 2021 | Zhang et al. | Yes | https://github.com/lllyasviel/SplitFilling |
| [Towards Vivid and Diverse Image Colorization with Generative Color Prior](https://arxiv.org/abs/2108.08826) | 2021 | Wu et al. | No | https://github.com/ToTheBeginning/GCP-Colorization |
| [Dual Color Space Guided Sketch Colorization](https://ieeexplore.ieee.org/abstract/document/9515572) | 2021 | Dou et al. | Yes | None |
| [Globally and Locally Semantic Colorization via Exemplar-Based Broad-GAN](https://ieeexplore.ieee.org/abstract/document/9566701) | 2021 | Li et al. | Yes | None |
| [Deep Edge-Aware Interactive Colorization against Color-Bleeding Effects](https://arxiv.org/abs/2107.01619) | 2021 | Kim et al. | Yes | https://github.com/niceDuckgu/CDR |
| [Bridging the Domain Gap towards Generalization in Automatic Colorization](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136770530.pdf) | 2022 | Lee et al. | No | https://github.com/Lhyejin/DG-Colorization |
| [ColorFormer: Image Colorization via Color Memory assisted Hybrid-attention Transformer](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136760020.pdf) | 2022 | Ji et al. | No | https://github.com/jixiaozhong/ColorFormer |
| [BigColor: Colorization using a Generative Color Prior for Natural Images](https://kimgeonung.github.io/bigcolor/) | 2022 | Kim et al. | No | https://github.com/KIMGEONUNG/BigColor |
| [Semantic-Sparse Colorization Network for Deep Exemplar-based Colorization](https://arxiv.org/abs/2112.01335) | 2022 | Bai et al. | Yes | https://github.com/bbaaii/SSC-Net |
| [DDColor: Towards Photo-Realistic Image Colorization via Dual Decoders](https://arxiv.org/abs/2212.11613) | 2022 | Kang et al. | No | https://github.com/piddnad/DDColor |
| [Lightweight Deep Exemplar Colorization via Semantic Attention-Guided Laplacian Pyramid](https://ieeexplore.ieee.org/abstract/document/10526459) | 2022 | Zou et al. | Yes | None |
| [UniColor: A Unified Framework for Multi-Modal Colorization with Transformer](https://arxiv.org/abs/2209.11223) | 2022 | Huang et al. | No | https://github.com/luckyhzt/unicolor |
| [Unsupervised Deep Exemplar Colorization via Pyramid Dual Non-Local Attention](https://ieeexplore.ieee.org/abstract/document/10183846) | 2022 | Wang et al. | Yes | https://github.com/wd1511/PDNLA-Net |
| [Improved Diffusion-based Image Colorization via Piggybacked Models](https://arxiv.org/abs/2304.11105) | 2023 | Liu et al. | No | https://github.com/hyliu/piggyback-color |
| [Two-Step Training: Adjustable Sketch Colourization via Reference Image and Text Tag](https://onlinelibrary.wiley.com/doi/full/10.1111/cgf.14791) | 2023 | Yan et al. | Yes | https://github.com/tellurion-kanata/sketch_colorizer |
| [Diffusing Colors: Image Colorization with Text Guided Diffusion](https://arxiv.org/abs/2312.04145) | 2023 | Zabari et al. | Yes | None |
| [Region Assisted Sketch Colorization](https://ieeexplore.ieee.org/abstract/document/10303276) | 2023 | Wang et al. | Yes | None |
| [L-CoIns: Language-based Colorization with Instance Awareness](https://openaccess.thecvf.com/content/CVPR2023/papers/Chang_L-CoIns_Language-Based_Colorization_With_Instance_Awareness_CVPR_2023_paper.pdf) | 2023 | Chang et al. | Yes | https://github.com/changzheng123/L-CoIns |
| [iColoriT: Towards Propagating Local Hint to the Right Region in Interactive Colorization by Leveraging Vision Transformer](https://arxiv.org/pdf/2207.06831) | 2023 | Yun et al. | Yes | https://github.com/pmh9960/iColoriT |
| [Adding Conditional Control to Text-to-Image Diffusion Models](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_Adding_Conditional_Control_to_Text-to-Image_Diffusion_Models_ICCV_2023_paper.pdf) | 2023 | Zhang et al. | Yes | https://github.com/lllyasviel/ControlNet |
| [L-CAD: Language-based Colorization with Any-level Descriptions using Diffusion Priors](https://arxiv.org/abs/2305.15217) | 2023 | Chang et al. | Yes | https://github.com/changzheng123/L-CAD |
| [Automatic Controllable Colorization via Imagination](https://arxiv.org/abs/2404.05661) | 2024 | Cong et al. | No | https://github.com/xy-cong/imagine-colorization |
| [Control Color: Multimodal Diffusion-based Interactive Image Colorization](https://arxiv.org/abs/2402.10855) | 2024 | Liang et al. | Yes | None |
| [Versatile Vision Foundation Model for Image and Video Colorization](https://dl.acm.org/doi/abs/10.1145/3641519.3657509) | 2024 | Bozic et al. | Yes | None |

### Bias Metrics

Various conceptualizations of bias have emerged in the image colorization space. Broadly, they include absolute metrics based on geometry, perceptual metrics based on non-uniformities in human color vision, and semantic metrics measuring how well the colorization preserves the semantic meaning of the image. Metrics can further be divided into those requiring a reference image and those that are automatic based on deep learning methods as explained in [Comparison of Metrics for Colorized Image Quality
Evaluation](https://www.vcl.fer.hr/papers_pdf/Comparison%20of%20Metrics%20for%20Colorized%20Image%20Quality%20Evaluation.pdf) (Žeger et al., 2022). As of December 2024, the two leading Python libraries for image quality assessment (IQA) include PyTorch Toolbox for Image Quality Assessment (PIQA) and PyTorch Image Quality (PIQ). Here, we provide a sampling of bias metrics over the years.

| Metric | Year | Type | Reference |
| - | - | - | - |
| Mean Squared Error (MSE) | - | Absolute | Yes |
| Mean Absolute Error (MAE) | - | Absolute | Yes |
| Peak Signal-to-Noise Ratio (PSNR) | - | Absolute | Yes |
| Kullback–Leibler Divergence (KL) | 1951 | Absolute | Yes |
| Earth Mover’s Distance (EMD) | 1989 | Absolute | Yes |
| CIEDE2000 | 2001 | Perceptual | Yes |
| Universal Image Quality Index (UIQI) | 2002 | Absolute | Yes |
| Structural Similarity Index Measure (SSIM) | 2004 | Perceptual | Yes |
| Visual Information Fidelity (VIF) | 2006 | Absolute | Yes |
| Feature Similarity Index Measure (FSIM) | 2011 | Perceptual | Yes |
| Naturalness Image Quality Evaluator (NIQE) | 2012 | Perceptual | No |
| Blind/Referenceless Image Spatial Quality Evaluator (BRISQUE) | 2012 | Perceptual | No |
| Gradient Magnitude Similarity Deviation (GMSD) | 2013 | Absolute | Yes |
| Learned Perceptual Image Patch Similarity (LPIPS) | 2018 | Semantic | Yes |
| Neural Image Assessment (NIMA) | 2018 | Perceptual | No |
| Deep Bilinear Convolutional Neural Network (DBCNN) | 2020 | Perceptual | No |
| Multi-Scale Image Quality Transformer (MUSIQ) | 2021 | Perceptual | No |

## Setup and Reproduction

This project uses Python 3.11.
1. ```git clone https://github.com/drakedu/colorful-bias```
2. ```pip install requirements.txt```
3. ```python download_data.py```
4. ```python sample_data.py```
5. ```python download_models.py```
6. ```python run_colorization.py```
7. ```python compute_metrics.py```
8. ```python analyze_colorization.py```

## Methods

### Choosing the Dataset

For this research project, [FairFace](https://github.com/dchen236/FairFace) was employed as the source of race/ethnicity-annotated facial images. This was because FairFace provides wide demographic coverage, including 9 distinct age groups, 7 race/ethnicity categories, and 2 gender groups, for 126 demographic subgroups. Additionally, the dataset is large enough that each individual age-gender-race subgroup contains at least 22 unique images, facilitating robust statistical analyses. Lastly, FairFace is openly accessible and does not require specific permissions, thereby facilitating the reproducibility and extension of this research project.

### Sampling Data

To facilitate consistency and comparability across demographic subgroups, we randomly sampled 22 images from each of the 126 demographic categories as defined by unique combinations of age, race/ethnicity, and gender, for a total of 2772 ground-truth images. By setting a baseline per subgroup, we ensured that every group was represented with an equal amount of data, mitigating potential issues arising from imbalanced class sizes.

### Downloading Models

While many colorization models have been proposed over the years, a sizeable proportion of them lack open-source implementations. For this research project, we spent 30 minutes attempting to set up each of 37 different models, of which 5 were successfully integrated. Due to time constraints, 9 other models received no attempt. Issues included datasets and pre-trained models no longer being publicly available, deprecated packages no longer being offered by channels such as Conda, stringent GPU requirements, macOS incompatibilities with LuaJIT and Caffe, intractable user input requirements, and domain limitations.

### Computing Metrics

To determine quality of image recolorization, we leveraged the PyTorch Toolbox for Image Quality Assessment due to its support of a wide-range of seminal metrics and ease of use. Out of the 38 metrics supported by the library, we were able to compute 28 of them for each of the 13860 image recolorings across the 5 models, possible largely by the help of 6 additional computers in Lamont Library and the blessing of library staff. Q-Align, AHIQ, TReS, MANIQA, ILNIQE, HyperIQA, BRISQUE, NRQM, and PI were projected to take 10 days, 5 days, 4 days, 3 days, 1 day, 16 hours, 15 hours, 7 hours, and 6 hours, respectively, while FID lacked a default image dataset fallback.

### Analyzing Colorization

After computing metrics between ground-truth images and reconstructions, we first explored differences across demographic groups using joyplots, facets, and CI-annotated barcharts. Next, we computed summary statistics for every age-gender-race-model combination, inclusive of an `All` keyword, to cover varying aggregations. We then conducted Welch's t-tests, checking for normality, and Mann-Whitney U tests to determine for each model whether there was a significant difference in its reconstruction scores by gender; by age through one-versus-rest; and by race in comparing White reconstructions against non-White reconstructions as well as against reconstructions from each non-White race/ethnicity. Next, we used one-way Welch's ANOVA tests after checking for normality to determine whether for each model there was a significant difference in its reconstruction scores by race and by age. We then use repeated-measures ANOVA tests after checking for normality to determine whether there was a significant difference in scores across models. Lastly, after checking for normality and homogeneity of variances, we use mixed-design ANOVA tests with model as the within-subject factor and race and age each as between-subject factors to determine if disparities by demographics vary across models. Due to issues involving numerical instability, sphericity was unable to be diagnosed for the repeated-measures ANOVA tests and the mixed-design ANOVA tests. Finally, throughout the analysis, normality checks were automated through Shapiro-Wilk tests as manual graphical analysis would not be tractable at this scale. Similarly, to test for homogeneity of variance for our mixed-design ANOVA, we leveraged Fligner-Killeen tests.

## Limitations

One of the limitations of our analysis was the relatively small sample size per demographic subgroup, as we only had access to 22 images per group. This restricted the statistical power of our tests, but future work can look at gaining access to larger annotated datasets. Additionally, the size of our suite of implemented models constrained our ability to draw conclusions about their differences in reconstruction disparities, though more models can be integrated going forward. Another challenge was the lack of both normality and homogeneity of variance in the data, each of which is important to many of the statistical tests we employed. Future work can thus investigate non-parametric methods that do not rely on these model assumptions. This would also mitigate the need to test for sphericity, which was unable to be checked in this research project due to numerical instability.

## Conclusion

Analyzing how colorful bias has changed over time brings us closer to understanding how we might proactively create systems and algorithms to combat it. From diversity in image datasets, knowledge of historical and cultural context, and conceptions of palatable color schemes, deconstructing exact sources of bias remains an open challenge as detailed in [The Limits of AI Image Colorization: A Companion](https://samgoree.github.io/2021/04/21/colorization_companion.html) (Goree, 2021). While new formulations for bias metrics such as those introduced in [Bias in Automated Image Colorization: Metrics and Error Types](https://arxiv.org/pdf/2202.08143) (Stapel et al., 2022) further complicate this endeavor, the increased focus on these normative questions in the space of image colorization in recent years brings hope for fairer and more inclusive technological progress.
