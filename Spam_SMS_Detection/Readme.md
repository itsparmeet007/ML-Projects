# Spam Analysis and Visualization

## Project Overview
This project performs a **data analysis of spam vs ham messages** and visualizes patterns in the dataset using **Python and Matplotlib**. The goal is to understand message distributions, identify common words in spam messages, and explore trends in the data.

This project focuses on **exploratory data analysis (EDA)** rather than advanced machine learning.

---

## Features

1. **Data Exploration**
   - Counts the number of spam and ham messages
   - Checks for missing values
   - Summarizes message lengths and other basic statistics

2. **Visualization**
   - Bar chart showing **spam vs ham distribution**
   - Word frequency charts to highlight **common words in spam and ham messages**
   - Histograms or boxplots to analyze message lengths

3. **Insights**
   - Identifies which words are most common in spam messages
   - Shows patterns in message length or frequency of certain words

---

## Project Structure
Spam-Analysis/
│
├── data/
│ └── spam.csv # Dataset with labeled messages
├── scripts/
│ └── spam_analysis.py # Python script for analysis and visualization
├── figures/
│ └── *.png # Generated plots (optional)
├── README.md # Project description (this file)
└── requirements.txt # Python dependencies

## 2. Run Analysis
python scripts/spam_analysis.py

Loads the dataset

Performs basic data exploration and statistics

Generates plots and visualizations

Saves plots in the figures/ folder or displays them

## Result 
<img width="607" height="543" alt="image" src="https://github.com/user-attachments/assets/1438a0c3-a31f-426c-97cd-92ede1745062" />

## Key Highlights

Simple and effective visualization of spam vs ham messages

Explores word frequency, message length, and distribution

Provides insights into patterns in spam messages

Fully modular and reusable for similar text datasets

## Future Enhancements

Add basic rule-based spam detection

Include interactive visualizations using Plotly or Seaborn

Expand dataset and perform statistical analysis on more features

Technologies Used

Python 3.x

Pandas, NumPy

Matplotlib for data visualization

## Author

## Parmeet – Student & Data Enthusiast

## A simple and insightful project for understanding spam message patterns and visualizing data effectively.


