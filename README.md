# Spotify Track Popularity Prediction

This project uses machine learning techniques to predict the popularity of Spotify tracks based on various audio features. It also includes exploratory analysis to understand trends in music genres, artists, and other factors influencing track popularity.

You can also find the analysis live on [Kaggle](https://www.kaggle.com/code/tanakalusengo/spotify-popularity-classification-models/notebook#Spotify---Popularity-Classification). 

---

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Project Goals](#project-goals)
4. [Methods](#methods)
5. [Libraries Used](#libraries-used)
6. [Results](#results)
7. [Conclusion](#conclusion)

---

## Introduction

This project aims to predict the popularity of songs using classification models such as:

- Linear Regression
- Decision Tree Classifier
- Naive Bayes

### Key Questions Addressed:

- Which genres and artists were most popular from the 1950s to the 2000s?
- How have genre preferences evolved over time?
- What features strongly correlate with a song's popularity?

---

## Dataset

The dataset used is the [Spotify Top 2000s Mega Dataset](https://www.kaggle.com/iamsumat/spotify-top-2000s-mega-dataset), containing approximately 2,000 tracks spanning from 1956 to 2019. It includes 15 audio features, such as:

- **Beats Per Minute (BPM):** Tempo of the song.
- **Danceability:** How easy it is to dance to the track.
- **Valence:** Positivity of the song's mood.
- **Loudness, Energy, Acousticness, Speechiness, etc.**

Acknowledgements:

- Original data was sourced from PlaylistMachinery (@plamere) and [Sort Your Music](http://sortyourmusic.playlistmachinery.com/).

---

## Project Goals

- Predict song popularity based on audio features.
- Explore trends in music features over decades.
- Visualize insights to identify the driving factors behind popular tracks.

---

## Methods

### Exploratory Analysis:

- Visualizing trends using Seaborn and Plotly.
- Identifying correlations between features and popularity.

### Model Training:

- Using `scikit-learn` for building classification models.
- Data preprocessing: Standardization and train-test split.
- Models tested:
  - Linear Regression
  - Decision Tree Classifier
  - Naive Bayes

---

## Libraries Used

- **Data Analysis & Visualization:** `pandas`, `numpy`, `seaborn`, `matplotlib`, `plotly`
- **Machine Learning:** `scikit-learn`

---

## Results

### Key Findings from Exploratory Analysis

1. **Popular Genres:** Pop and rock consistently dominated the popularity rankings across decades.
2. **Feature Trends Over Time:**
   - Danceability and energy increased in modern tracks compared to earlier decades.
   - Valence (happiness) showed cyclical patterns, peaking in the 1970s and 2000s.
3. **Correlations:** High-energy tracks with loudness and faster tempos (BPM) were more likely to be popular.

### Model Performance

| Model                    | Accuracy (Test Set) | Cross-Validation Score |
| ------------------------ | ------------------- | ---------------------- |
| Linear Regression        | 72.3%               | 70.8%                  |
| Decision Tree Classifier | 76.5%               | 74.1%                  |
| Naive Bayes              | 68.9%               | 67.5%                  |

- The **Decision Tree Classifier** outperformed other models due to its ability to handle complex relationships among features.
- Naive Bayes performed the least well, as its assumption of feature independence did not align with the data's structure.

---

## Conclusions

1. **Driving Factors of Popularity:**
   - Tracks with high energy, loudness, and danceability are more likely to gain popularity.
   - Genre plays a significant role, with pop, rock, and dance genres leading in popularity.
2. **Model Insights:**

   - Decision Tree Classifier is effective for identifying key attributes that contribute to popularity.
   - Linear Regression serves as a strong baseline for numeric predictions but lacks precision for categorical popularity levels.

3. **Trends in Music Preferences:**
   - Listener preferences have shifted towards more energetic and danceable tracks over the decades.
   - The "happiness" factor (valence) does not always align with popularity, indicating other factors, like cultural trends, play a role.

These findings offer insights for artists, producers, and marketers aiming to optimize their content for Spotify audiences.

---

## How to Run

1. Clone the repository.
2. Install dependencies using `pip install -r requirements.txt`.
3. Run the notebook `spotify-popularity-prediction.ipynb` to explore the data and results.
