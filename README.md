# Text-Classification
Implement some state-of-the-art text classification models with TensorFlow.

## Requirement

- Python3
- TensorFlow >= 1.4

Note: Original code is written in TensorFlow 1.4, while the `VocabularyProcessor` is depreciated, updated code changes to use `tf.keras.preprocessing.text` to do preprocessing. The **new** preprocessing function is named `data_preprocessing_v2`

## Dataset

You can load the data with

```python
dbpedia = tf.contrib.learn.datasets.load_dataset('dbpedia', test_with_fake_data=FLAGS.test_with_fake_data)
```

## Attention is All Your Need

Paper: [Attention Is All You Need](http://arxiv.org/abs/1605.07725)

See multi_head.py

Use self-attention where **Query = Key = Value = sentence after word embedding**

Multihead Attention module is implemented by [Kyubyong](https://github.com/Kyubyong/transformer)

## IndRNN for Text Classification

Paper: [Independently Recurrent Neural Network (IndRNN): Building A Longer and Deeper RNN](https://arxiv.org/abs/1803.04831)

IndRNNCell is implemented by [batzener](https://github.com/batzner/indrnn)

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


## Convolutional Neural Networks for Sentence Classification

Paper: [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882)

See: cnn.py


## RMDL: Random Multimodel Deep Learning for Classification

Paper: [RMDL: Random Multimodel Deep Learning for Classification](https://arxiv.org/abs/1805.01890)

See: RMDL.py
See: [RMDL Github](https://github.com/kk7nc/RMDL)




**Note**: The parameters are not fine-tuned, you can modify the kernel as you want.
## Performance

| Model                               | Test Accuracy | Notes                   |
| ----------------------------------- | ------------- | ----------------------- |
| Attention-based Bi-LSTM             | 98.23 %       |                         |
| HAN                                 | 89.15%        | 1080Ti 10 epochs 12 min |
| Adversarial Attention-based Bi-LSTM | 98.5%         | AWS p2 2 hours          |
| IndRNN                              | 98.39%        | 1080Ti 10 epochs 10 min |
| Attention is All Your Need          | 97.81%        | 1080Ti 15 epochs 8 min  |
| RMDL                                | 98.91%        | 2X Tesla Xp (3 RDLs)    |
| CNN             | To be tested      | To be done  |

## Welcome To Contribute

If you have any models implemented with great performance, you're welcome to contribute. Also, I'm glad to help if you have any problems with the project,  feel free to raise a issue.



