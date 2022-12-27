# Ceiling Analysis

If there is a pipeline such as:

```bash
image --> text detection --> char segmentation --> char recognition
```

...we would have an overall accuracy. By modifying the input of each step in
the pipeline with perfect input data (starting with the first one)
we can get what the possible accuracy improvement per step is.

| Component         | Accuracy | Potential |
|------------------------------|-----------|
| overall system    | 72%      |           |
| text detection    | 89%      | 17%       |
| char segmentation | 90%      | 1%        |
| char recogintion  | 100%     | 10%       |
