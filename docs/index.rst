
BENDeep
=======

``BENDeep`` is a pytorch based deep learning solution for Bengali NLP Task like ``bengali translation``\ , ``bengali sentiment analysis`` and so on. 

Installation
------------

``pip install bendeep``

Dependency
^^^^^^^^^^


* pytorch 1.5.0+

Pretrained Model
----------------


* `Sentiment Analysis <https://github.com/sagorbrur/bendeep/tree/master/models/sentiment>`_
* `Translation Model <https://github.com/sagorbrur/bendeep/tree/master/models/translation>`_

API
---

Sentiment Analysis
^^^^^^^^^^^^^^^^^^

Analyzing Sentiment
~~~~~~~~~~~~~~~~~~~

This sentiment analysis model is a RNN based ``GRU`` model trained with `socian sentiment dataset <https://github.com/socian-ai/socian-bangla-sentiment-dataset-labeled>`_ with loss 0.073 in 150 epochs.
Dataset size: 4000 sentences

.. code-block:: py

   from bendeep import sentiment
   model_path = "senti_trained.pt"
   vocab_path = "vocab.txt"
   text = "আজকের আবহাওয়া খুব সুন্দর।"

   sentiment.analyze(model_path, vocab_path, text)

Training Sentiment Model
~~~~~~~~~~~~~~~~~~~~~~~~

To train this model you need a csv file with one column ``review`` means text and another column ``sentiment`` with 0 or 1, where 1 for positive and 0 for negative sentiment.

Example:

.. code-block::

   ,review,sentiment
   0,তোমাকে খুব সুন্দর লাগছে।,1
   1,আজকের আবহাওয়া খুব খারাপ।,0

.. list-table::
   :header-rows: 1

   * - 
     - review
     - sentiment
   * - 0
     - তোমাকে খুব সুন্দর লাগছে।
     - 1
   * - 1
     - আজকের আবহাওয়া খুব খারাপ।
     - 0


.. code-block:: py

   from bendeep import sentiment
   data_path = "sentiment_data.csv"
   sentiment.train(data_path)
   # you can also pass these parameter
   # sentiment.train(data_path, batch_size = 64, epochs=100, model_name="trained.pt")

after successfully training it will complete training and save model as ``trained.pt`` also save vocab file as ``vocab.txt``

Machine Translation
^^^^^^^^^^^^^^^^^^^

Translate Bengali to English
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This model is a seq2seq attentional model trained with `this <https://github.com/sagorbrur/bendeep/tree/master/data>`_ dataset with loss 0.0.

.. code-block:: py


   from bendeep import translation
   from bendeep.translation import EncoderRNN
   from bendeep.translation import AttnDecoderRNN

   data_path = "data/translation/eng-ben.txt"
   encoder = "models/translation/encoder.pt"
   decoder = "models/translation/decoder.pt"
   input_sentence = "আমার শীত করছে।"
   translation.bn2en(data_path, encoder, decoder, input_sentence)
   # outupt
   # > আমার শীত করছে ।
   # = i feel cold .

Training Translation Model
~~~~~~~~~~~~~~~~~~~~~~~~~~

To train translation model you need a dataset in ``.txt`` format with tab separate ``input`` and ``target`` sentences.

Example:

.. code-block::

   I eat rice. আমি ভাত খাই।
   He goes to school.  সে বিদ্যালয়ে যায়।

.. code-block:: py

   from bendeep import translation
   from bendeep.translation import EncoderRNN
   from bendeep.translation import AttnDecoderRNN

   data_path = "data/translation/eng-ben.txt"
   translation.training(data_path, iteration=75000)

after successfully training it will complete training and save encoder and decoder model as ``encoder.pt``\ , ``decoder.pt``. Also display some random evaluation results.

References
----------


* `pytorch <https://pytorch.org/>`_
* `pytorch tutorial <https://pytorch.org/tutorials/>`_
* `en-bn dataset <https://www.manythings.org/anki/>`_
* `socian sentiment dataset <https://github.com/socian-ai/socian-bangla-sentiment-dataset-labeled>`_

