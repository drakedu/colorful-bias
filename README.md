# Colorful Bias

In 2015, Google Photos faced widespread backlash after its [algorithms mislabeled](https://www.nytimes.com/2023/05/22/technology/ai-photo-labels-google-apple.html) Black people as gorillas (Grant & Hill, 2023). Three years later, the MIT Media Lab found that facial recognition systems had [error rates](https://www.media.mit.edu/articles/study-finds-gender-and-skin-type-bias-in-commercial-artificial-intelligence-systems/) as high as 34% for darker-skinned women compared to less than 1% for lighter-skinned men (Buolamwini, 2018). From image classification to facial recognition, computer vision is infamously flawed. In this research project, I investigated how these issues of fairness manifest in the age of generative AI. In particular, I explored the robustness of generative algorithms for image colorization with respect to skin tone bias. To accomplish this, I conducted a survey of race/ethnicity-annotated face datasets, compiled seminal algorithms for image colorization over the years, researched various formulations of bias metrics, and set up a code framework with statistical tests to rigorously compare the performance of coloring procedures. Through the above work, I sought to shed light on the trend in “colorful” bias, or bias in algorithmic colorization of images containing human skin tones, as seen through algorithms over time.

## Race/Ethnicity-Annotated Face Datasets

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
