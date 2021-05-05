# model_serving_test
## Model server (Flask):
```python test_v2.py```
## Model server (FastAPI):
```python test_v3.py```
## To test:
```
from request_module import Serving_clf
requestor = Serving_clf(url=__INSERT URL HERE__)
prediction = requestor.predict([frame])
```
where frame is an RGB image array.

prediction should return a json in the form of:
```json
{
  'predictions': {
    'points': ([l,t,r,b]),
    'attributes-value': (text)
  }
}
```
