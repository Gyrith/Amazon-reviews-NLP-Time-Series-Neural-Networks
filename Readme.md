# Amazon Reviews Analysis: NLP, Time Series and Neural Networks

Summative lab for an analytics-firm client scenario. One notebook analyses a dataset of 997 Amazon customer reviews in three parts: text analysis, review-volume forecasting and star-rating prediction.

## Contents

| File | Description |
|---|---|
| `C09_MO8.ipynb` | The full analysis, with code, outputs and a written reflection for each part |
| `amazon_reviews_lab.csv` | The dataset (997 reviews, May 2000 to July 2014) |
| `C09_MO8L.ipynb` | Starter notebook |

### Dataset columns

`product_id`, `rating` (1-5), `review_text`, `review_summary`, `review_time_raw`, `unix_review_time`, `timestamp`, `sentiment_label`, `review_length_words`

Most reviews are for e-readers and tablets (Nook, Kindle), plus TV mounts and cables. 58% of reviews are 5 stars.

## What the notebook does

### Part 1: NLP
- **Preprocessing:** lowercasing, URL and punctuation removal, tokenization, stopword removal and lemmatization. The negation words "not", "no" and "nor" are deliberately kept, because removing them would make negative reviews look positive.
- **Validation:** vocabulary shrinks from 7,532 to 6,563 words, with no reviews left empty.
- **Exploration:** word, bigram and trigram frequencies, top words by sentiment group, TF-IDF distinctive terms and word clouds.

### Part 2: Time series
- **Series:** monthly review count and average rating, 2010 to 2014.
- **Stationarity:** ADF and KPSS tests, log transform, differencing and seasonal decomposition. Both series are non-stationary and become stationary after one difference (d = 1).
- **Modeling:** ACF/PACF plots, an ARIMA grid search compared by AIC and BIC, and a 6-month hold-out forecast scored with MAE and RMSE. ARIMA(1,1,1) had the lowest AIC (MAE 10.2, RMSE 11.9). ARIMA(1,0,0) forecast the hold-out best (MAE 3.6, RMSE 4.1).
- **Diagnostics:** Ljung-Box test on the residuals (p = 0.57 at lag 6, p = 0.42 at lag 12), so no leftover autocorrelation.

### Part 3: Neural networks
- **Task:** predict the 1-5 star rating from review text.
- **Features:** TF-IDF on 2,000 unigrams and bigrams, with the vectorizer fit on the training split only.
- **Model:** a multi-layer perceptron built with TensorFlow/Keras: Dense 64 (ReLU), Dropout 0.5, Dense 32 (ReLU), Dropout 0.3 and Dense 5 (softmax), trained with Adam, class weights and early stopping.
- **Evaluation:** accuracy against a majority-class baseline, confusion matrix, classification report, and a comparison of optimizers (Adam, RMSprop, SGD), class weighting and dropout.

## Key results

| Result | Value |
|---|---|
| Test accuracy (main model) | 54.5% |
| Majority-class baseline (always predict 5 stars) | 58% |
| Macro-F1, unweighted Adam vs class-weighted Adam | 0.157 vs 0.311 |
| Best F1 by rating | 5 stars (0.72), then 1 star (0.40) |
| Hardest ratings | 2 and 3 stars (F1 0.13 and 0.14) |

The network overfits (0.932 training accuracy against 0.544 validation accuracy for class-weighted Adam) and favors the 5-star class. Accuracy alone is misleading here because of the class imbalance, so macro-F1 and the confusion matrix matter more. See the reflection at the end of each part for the full discussion.
