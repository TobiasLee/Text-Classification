# Text-Classification
Implement some state-of-the-art text classification models with TensorFlow.

## Attention is All Your Need

Paper: [Attention Is All You Need](http://arxiv.org/abs/1605.07725)

See all_attention.py

Use self-attention where **Query = Key = Value = sentence after word embedding**

Multihead Attention module is implemented by [Kyubyong](https://github.com/Kyubyong/transformer)

## IndRNN for Text Classification

Paper: [Independently Recurrent Neural Network (IndRNN): Building A Longer and Deeper RNN](https://arxiv.org/abs/1803.04831)

IndRNNCell is  implemented by [batzener](https://github.com/batzner/indrnn)

## Attention-Based Bidirection LSTM for Text Classification

Paper: [Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification](http://www.aclweb.org/anthology/P16-2034)  

See attn_bi_lstm.py

## Hierarchical Attention Networks for Text Classification

Paper: [Hierarchical Attention Networks for Document Classification](http://aclweb.org/anthology/N16-1174)

See attn_lstm_hierarchical.py

Attention module is implemented by [ilivans/tf-rnn-attention ](https://github.com/ilivans/tf-rnn-attention).

## Adversarial Training Methods For Supervised Text Classification

Paper: [Adversarial Training Methods For Semi-Supervised Text Classification](http://arxiv.org/abs/1605.07725)

See: adversrial_abblstm.py


## Dataset

You can load the data with

```python
dbpedia = tf.contrib.learn.datasets.load_dataset('dbpedia', test_with_fake_data=FLAGS.test_with_fake_data)
```

## Performance

| Model                               | Accuracy     | Notes                                    |
| ----------------------------------- | ------------ | ---------------------------------------- |
| Attention-based Bi-LSTM             | 98.23 %      |                                          |
| HAN                                 | To be tested |                                          |
| Adversarial Attention-based Bi-LSTM | 98.5%        | AWS p2 2 hours                           |
| IndRNN                              | To be tested |                                          |
| Attention is All Your Need          | 95.650792 %  | Train 10 epochs, using 10% training data |

## TO DO
1. Test models performance
2. Code refactoring







