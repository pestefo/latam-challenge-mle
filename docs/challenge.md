# Challenge Documentation

Author: Pablo Estefó

Date: 19/12/2023

## Part I

### Transcription of the notebook into the api implementation

I found a couple if minor mistakes in the syntax related to calling functions with keyword arguments only in some of the arguments and not all of them. Example

```python
# Original
sns.barplot(data.index, data.values, color='lightblue', alpha=0.8)
# Fix
sns.barplot(x=data.index, y=data.values, color='lightblue', alpha=0.8)
```

I found also that the `training_data` defined in **4.a** was not used, ergo, it was not used when training the models.

As the data scientist is responsible for the analysis, its background rationale and completeness are trustworthy.

### Choosing a model

The performance given by the models (balanced and featured optimized) was:

***XGBoostClassifier***

| class | precision | recall | f1-score | support |
| --- | --- | --- | --- | --- |
| 0 | 0.88  | 0.52 | 0.65 | 18403 |
| 1 | 0.24 | 0.68 | 0.36 | 4105 |

***LogisticRegression***

| class | precision | recall | f1-score | support |
| --- | --- | --- | --- | --- |
| 0 | 0.87 | 0.59 | 0.7 | 18403 |
| 1 | 0.25 | 0.61 | 0.35 | 4105 |

I talked with an ex-employee for LATAM to understand the severity of False Negatives and False Positives. I understood that False Negatives (not identifying a delayed flight) have the most negative consequences on the business side: client experience and budget-for-delays are highly impacted. A high amount of False Positives would mean a more conservative approach which have to be acknowledged by the stakeholders as it may increase the emergency budget unnecessarily. 

Given that analysis, we should care the most of the value of ************recall for class 1************. The results show that the XGBoostClassifier performs slightly better than LogisticRegression, but, as this metric is critical for the reliability of the system, the first model is the chosen one.

In addition, we have to consider that XGBoostClassifier typically takes more time and computational resources for training than LogisticRegression. In such case, the historical data have to be well curated that a high quality and small enough to speed up this process without affecting its recall performance. This also has to be taken into account when scheduling training the model.

Another relevant aspect is explainability. LogisticRegression is a less obscure model than XGBoostClassifier. The chosen model behaves more like a “black box” and we only rely on the perfomance metrics.

In conclusion, we choose the XGBoostClassifier because of its ability to detect False Negatives which are the most impactful cases in the operation.

### Technical notes on the implementation

The implementation of the `DelayModel` class respected the given template of the public methods: `preprocess`, `fit` and `predict`.

I implemented a cache for the `XGBoostClassifier` instance, in order to not to train the model every time  `DelayModel` class is instantiated. 

We can highlight the separation of concerns of the different private methods that encapsulate context-related statements for improving maintainability and readability. Also, I use descripive names and every method ios werll documented

I made a correction in the `test_model_fit` of the `test_model.py` . In the fourth statement it calls the `predict` function directly from the private attribute `_model` instead of using the public method `DelayModel::predict` as it should. It is wrong because It breaks the encapsulation OOP principle and it does not align with the purpose of the test battery. The fix is the following:

```python
# Original
predicted_target = self.model._model.predict(features_validation)
# Fix
predicted_target = self.model.predict(features_validation)
```

## Part II

For the api implementation the work was divided in two parts: make use of a `DelayModel` instance to handle the hits to the `/predict` endpoint, and the validation of the requests’ payload.

For the first part, there is not much to say: an instance of `DelayModel` is available for all the endpoints.

For the second part, I made use of the `Pydantic` library that is the recommended library for data model validation. We modeled the requests’ data using the `FlightListModel` which is a list of `FlightModel`. This model has validations for the inputs, for example: `TIPOVUELO` attribute is a `string` that accepts the values `"N"` or `"I"` only. Both the input and the output are handled using these pydantic models.

I added the `Fecha_I` and `Fecha_O` keys for valid requests as they are needed when adding the additional attributes in the `DelayModel::preprocess` method. They they match to the `Fecha-I` and `Fecha-O` data columns respectively.

I fixed a typo in the tests `test_..._unkown_...` for `test_..._unknown_...`. I also added two tests for testing: `Fecha-I`/ `Fecha-O` keys (`test_should_failed_unknown_fecha`), and one for testing predicting more than one flight (`test_should_get_predict_more_than_one_flight`).

## Part III

I deployed the API in a AWS EC2 instance that can be accessed through the URL: `http://54.234.143.225/[ENDPOINT]`

It was tested through POSTMAN requests (see the `./docs/LATAM-Challenge-Requests.postman_collection.json` file)

## Part IV

Not addressed.