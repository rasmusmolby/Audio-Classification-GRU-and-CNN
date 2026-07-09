# Audio Classification using a hybrid GRU and CNN model.

Classify stationary, transient and disturbing human blabber for an active noise cancelling problem. 
This model was made with a big focus on being memory and storage effecient, as its proof of concept is to run on a chip in headsets, that is limited to only 64 KB of memory.
This is only for experimental use, and is not used in any commercial setting.
Therefore, this is not an optimal setup for this problem, and many aspects probably wont work correctly or are not explained or described as they should be

# What this project set out to do
This setup was to classify 3 audio types (Transient, Static, and People talking) with a machine learning algortihm, that focus on minimalizing the memory consumptions as much as possible.
As this was a school project that colaborated with RTX, we were mainly experimenting with deep learning models rather than old school machine learning.

# What we did
We have researched multiple models. miniTransformers, CNN, CRNN, LSTM, GRU, and multiple hybrids of the mentioned models.
The best performing model in regards to the tradeoff of accuracy and memory constraints were a GRU-CNN hybrid.

# Datasets
Mainly used a combination of ESC-50, Librispeech, P.50, and Urbansounds 8k.

# For further research
View branches and/or dm for final report papers.

This work is made under Aalborg University, and should therefore only be viewed as a research project with no intentions of commercial use.
