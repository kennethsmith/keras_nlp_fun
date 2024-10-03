# keras_nlp_fun

## purpose
Learn keras_nlp a bit. 

## references
* https://keras.io/guides/keras_nlp/getting_started/

## data
    # Dir paths running from the app directory:
    # data_paths = [
    # '../data/aclImdb/train',
    # '../data/aclImdb/test',
    # ]

    !curl -O https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
    !tar -xf aclImdb_v1.tar.gz
    !# Remove unsupervised examples, num_classes value will cause an error if unsup is not removed.
    !rm -r aclImdb/train/unsup

# notes
- tensorflow-metal wouldn't work for the fitting so training was slow.
- I set epochs to 1 to save time, obviously more would yield better results.
- There were some other tweaks I made to make the code from the website run right.
- Lots of warnings...