## UNSW-NB15 Overview

UNSW-NB15 is a contemporary dataset created as part of the UNSW Cyber Security Research Centre. It is designed for evaluating intrusion detection systems and network security. The dataset contains a diverse range of network traffic and attack scenarios.

### Download

The UNSW-NB15 dataset can be downloaded from the official website:
- [UNSW-NB15 Dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset)

For the direct download link:
- [Direct Download Link](https://unsw-my.sharepoint.com/:f:/g/personal/z5025758_ad_unsw_edu_au/EnuQZZn3XuNBjgfcUu4DIVMBLCHyoLHqOswirpOQifr1ag?e=gKWkLS)

Select `CSV Files` then choose the four CSV files to get the dataset in CSV format.

Or:

In Ubuntu/Linux terminal:

```bash
mkdir Dataset && cd Dataset

wget https://unsw-my.sharepoint.com/personal/z5025758_ad_unsw_edu_au/_layouts/15/download.aspx?UniqueId=4c4b6d3a%2Dd264%2D487b%2Db677%2Ddfac2175f783 -O UNSW-NB15_1.csv
wget https://unsw-my.sharepoint.com/personal/z5025758_ad_unsw_edu_au/_layouts/15/download.aspx?UniqueId=cb831326%2De652%2D459d%2D8033%2Df773d42f60ec -O UNSW-NB15_2.csv
wget https://unsw-my.sharepoint.com/personal/z5025758_ad_unsw_edu_au/_layouts/15/download.aspx?UniqueId=1dd6a5d4%2Dab33%2D4fe6%2D838f%2Dfe16125f744f -O UNSW-NB15_3.csv
wget https://unsw-my.sharepoint.com/personal/z5025758_ad_unsw_edu_au/_layouts/15/download.aspx?UniqueId=5b2fa2fa%2D89b0%2D494d%2D8495%2D1df77ae0f259 -O UNSW-NB15_4.csv
```

### Key Features:
- **Variety of Attacks**: Includes **nine** different classes of attacks such as DoS, backdoors, worms, and more.
- **Large Volume**: Comprises over **2.5 million** records, allowing for extensive testing and validation.
- **Detailed Record Structure**: Each record contains **49** attributes, providing in-depth information for analysis.
- **Synthetic Generation**: Data generated using a hybrid approach combining real and synthetic traffic to mimic real-world usage.

### Applications:
- Development and testing of intrusion detection systems
- Machine learning and AI-based security research
- Network performance evaluation under attack conditions

The UNSW-NB15 dataset is an essential tool for advancing research in cybersecurity and enhancing the effectiveness of IDS solutions.