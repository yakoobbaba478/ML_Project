# Deep learning based recommondation system for apparels.

about
=====
Build a recommendation system based on deep learning techniques.A system which accepts images of apparels,
classify the apparel type and do a feature vector search to recommend similar type apparels.
Data-set is created by scraping several websites(non-commercial use)
and applied some augmentations to the data-set to increase the size of the data.
Had a VVG16 classifier to classify apparel type(ex. shirt,trouser,etc.).
Recommend related apparels from same class by feature vector search computed using VGG16 network over available apparel list.
