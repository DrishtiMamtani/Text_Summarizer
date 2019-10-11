# Text_Summarizer


In today’s world, we enjoy quick access to enormous amounts of information, most of which are redundant and may not convey the intended meaning. Therefore, it is necessary to summarize this information. 
Text summarization is the technique for generating a concise and precise summary of voluminous texts while focusing on the sections that convey useful information, and without losing the overall meaning. The focus is to transform the lengthy documents into shortened versions. Manually generating a summary can be time-consuming and tedious.
Deep learning algorithms were used to develop the text summarizer.



Overview:

“Automatic text summarization is the task of producing a concise and fluent summary while preserving key information content and overall meaning”. Text summarization is the problem of creating a short, accurate, and fluent summary of a longer text document. Abstractive machine learning algorithms are better but these require complicated deep learning techniques and sophisticated language modeling. Beautiful Soup Library is used to fetch the data and Keras and Tensorflow are used to use Tokenizer and LSTM. A dataset is selected and text cleaning and summary cleaning is performed. The data is divided into training and testing data using the holdout method. The sentences in the training data are tokenized and stacked LSTM is used to build the model. After around 10 epochs model will be validated on the training set to calculate accuracy and Precision.

Scope and necessity of the system:

Currently, there is a huge amount of data circulating in the digital space out of which most of the information is redundant and does not convey the required meaning. For example, if we are looking for some specific information regarding any topic online, we may have to invest a lot of time in discarding the unnecessary data before getting our required information.
Hence, there arises a need to design a machine learning algorithm that can provide us an accurate summary of the long text given. 
Today's computers are a lot faster than the human mind and it is most likely that the computer would create a more precise summary before humans can. It has also been observed that automatic summary software is capable of summarizing texts of 500 words in a split second for which humans would require at least 15 minutes. Also, the software can be used to summarize the webpages which speed up our surfing process and shrink the original article to around 20% of the original article. Using the summarizer ensures that all the important facts are mentioned which can be overlooked by humans.
Hence, using the text summarizer reduces our reading time, time spent in researching for information, and increases the amount of information that can fit in an area.


Feasibility Report:

Encoder and Decoder architecture  is used in Training as well as Testing Phase. 

Training Phase:
An Encoder Long Short Term Memory model (LSTM) reads the entire input sequence wherein, at each timestep, one word is fed into the encoder. It then processes the information at every timestep and captures the contextual information present in the input sequence. The hidden state (hi) and cell state (ci) of the last time step are used to initialize the decoder.
The decoder is also an LSTM network that reads the entire target sequence word-by-word and predicts the same sequence offset by one timestep. The decoder is trained to predict the next word in the sequence given the previous word. <start> and <end> are the special tokens that are added to the target sequence before feeding it into the decoder. 
Testing Phase:
After training, the model is tested on new source sequences for which the target sequence is unknown. So, we need to set up the inference architecture to decode a test sequence.
Firstly Encode the entire input sequence and initialize the decoder with internal states of the encoder. Then Pass <start> token as an input to the decoder. Run the decoder for one timestep with internal states. The output will be the probability for the next word. The word with the maximum probability will be selected. Pass the sampled word as an input to the decoder in the next time step and update the internal states with the current time step.
Global Attention:
Instead of looking at all the words in the source sequence, we can increase the importance of specific parts of the source sequence that result in the target sequence. This is the basic idea behind the attention mechanism. In Global attention, attention is placed on all the source positions. In other words, all the hidden states of the encoder are considered for deriving the attended context vector.



Literature Study:

There have been various approaches for implementing an Automatic Text Summarizer. In the start researchers only focused on one component of sentence significance, namely, the presence of frequently occurring keywords. However, in the late 1950s in a paper published by Hans Peter Luhn, he used word frequency, phase frequency,  heading words, and structural indicators for the extraction of vital information from a given text.
There are broadly two different approaches that are used for text summarization namely Abstractive and Extractive. Extractive text summarization involves the selection of phrases and sentences from the source document to make up the new summary. Abstractive text summarization involves generating entirely new phrases and sentences to capture the meaning of the source document.
Extractive text summarization is implemented using a text rank algorithm. It is an extractive and unsupervised text summarization technique.


To begin with, we merge all the text in the articles and then split the text into individual sentences. In the next step, we find the sentence vectors and compute a similarity matrix to store the similarities between the calculated vectors. Further, we convert the matrix into a graph with sentences as vertices and similarity scores as edges, for sentence rank calculation. Finally, a certain number of top-ranked sentences form the final summary.
The other approach is the abstractive one which has been discussed in detail in the design document.


Conclusion:

Automatic Text Summarization is a hot topic of research. Moving on, we will build an abstractive summarization with the help of deep learning algorithms.
The summarizer will be a single document summarizer and the language of the summary depends on the training dataset. RNNs and LSTM will be used to carry out the task. Encoder- decoder architecture and attention mechanism play a vital role in building the summarizer. 

