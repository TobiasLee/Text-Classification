# Text-Classification
Implement two text classification with TensorFlow based on two papers:

## Attention-Based Bidirection LSTM for Text Classification

Paper: [Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification](http://www.aclweb.org/anthology/P16-2034)  —— Zhou, Peng

See ABBLSTM.py

## Hierarchical Attention Networks for Text Classification

Paper: [Hierarchical Attention Networks for Document Classification](http://aclweb.org/anthology/N16-1174)

See ALSTM_Hierarchical.py

## Dataset and Performance

You can load the data with

```python
 dbpedia = tf.contrib.learn.datasets.load_dataset('dbpedia', test_with_fake_data=FLAGS.test_with_fake_data)
```

The ABBLSTM model can get a 98.23 % accuracy on the dataset.







