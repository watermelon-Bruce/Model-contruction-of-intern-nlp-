# How to run the code

open the Divide_machine.py file in the directory core, in which the whole work flow was displayed.(note: you can change the config.yaml file to satisfy your own demand)

# work flow

![image text]([Model-contruction-of-intern-nlp-/work flow.png at main · watermelon-Bruce/Model-contruction-of-intern-nlp- (github.com)](https://github.com/watermelon-Bruce/Model-contruction-of-intern-nlp-/blob/main/work flow.png))

# project structure

```
├── Readme.md                               // help
├── core                                        
|  ├── Divide_machine.py                    // entrance of the program
├── set                                         
|  ├── config.yaml                          // config file
|  ├── config.py                            // config function
├── utils                         
|  ├── get_union.py                         // union the keywords
|  ├── word_cut.py                          // cut the sentence to word
|  ├── get_word_vec.py                      // use word_to_vec model to get word vectors
|  ├── get_doc_vec.py                       // get doc vectors
|  ├── get_doc_tfidf.py                     // get doc tfidf vectors
|  ├── get_train_validation.py              // get validaiton and train data
|  ├── get_kernel_vec.py                    // clusting to get doc embedding with stronger features
|  ├── navie_bayes_divider.py               // naive base divider
|  ├── svm_divider.py                       // svm divider
|  ├── re_sampling.py                       // sampler(inculding upsampling and downsampling in order to balance data)
|  ├── get_accuracy_and_recall.py           // calculate accuracy and recall
|  ├── model_read.py                        // read multiple divider
|  ├── consin_similarity.py                 // calculate similarity
|  ├── consin_similarity_average.py         // calculate average similarity between one vector and the others 
|  ├── get_category.py                      // get the category of ducuments
|  ├── get_category_list_by_industry.py     // 
|  ├── get_encode.py                        // get the encoding of file
|  ├── get_group_dir.py                     // create directory under specific path
|  ├── get_stoplist.py                      // get stopword list
|  ├── open_file.py                         // open multiple kinds of files
|  ├── pickle_operation.py                  // read and store pickle files
├── docs                                        
|  ├── file                                     
|  |  ├── chen_phrase_final                 // phrases got by chen
|  |  ├── yang_phrase_final                 // phrase got by yang
|  |  ├── total_phrase_final                // union of stopwords
|  ├── total_content                        // all the txt content
|  ├── total_seglist                        // all the word cutting result
|  ├── WordToVec                            // the model of wordtovec as well as vector matrix
|  ├── DocVec                               // docembedding
|  ├── DocVec_train                         // traning data of docembedding
|  ├── DocVec_validation                    // validation data of docemebdding
|  ├── DocVec_kernel                        // docembedding with strong features
|  ├── DocTfidf                             // tfidf matrix
|  ├── DocTfidf_train                       // training data of tfidf
|  ├── DocTfidf_validation                  // validation data of tfidf
|  ├── DocVec_kernel                        // tfidf with strong features
|  ├── navie_bayes_model                    // multiple naive bases dividers based on docembedding
|  ├── navie_bayes_model_tfidf              // multiple naive bases dividers based on tfidf
|  ├── svm_model                            // multiple svm dividers based on docembedding
|  ├── svm_model_tfidf                      // multiple svm dividers based on tfidf
|  ├── predict_result                       // prediction result after voting
|  ├── stopwords                            // stopwords
```

# Modules

## data_processor.To_union()

### function：

```
get the union set of keywords
```

### parameters:

```
chen_phrase_set_final:phrases got by chen
yang_phrase_set_final：phrases got by yang
total_phrase_set_final：path for the union set
```

## data_processor.To_word_cut()

### function：

```
cut word
```

### parameters:

```
stop_words_road：path of stopwords
total_phrase_set_final:path for the union set of keywords
total_content_by_industry_dir_road:path of the content txt file
total_seglist_by_industry_dir_road：path of word cutting result
```

## data_processor.To_word_vec()

### function：

```
get word embedding
```

### parameters:

```
total_seglist_by_industry_dir_road:path of word cutting result
word_vec_model：the model of wordtovec as well as vector matrix
stop_words_road: path of stopwords
```

## data_processor.To_doc_vec()

### function：

```
get doc embedding
```

### parameters:

```
word_vec_model:the model of wordtovec as well as vector matrix
total_phrase_set_final:path for the union set of keywords
total_seglist_by_industry_dir_road:path of word cutting result
doc_vector：path of doc embedding 
```

## data_processor.To_doc_tfidf()

### function：

```
get tfidf vector for documents
inculding the svd decompositon of tfidf matrix and adjustment of tfidf matrix by keywords
```

### parameters:

```
total_phrase_set_final:path for the union set of keywords
total_seglist_by_industry_dir_road:path of word cutting result
doc_tfidf：path of tfidf matrix
```

## data_processor.To_train_validation()

### function：

```
get training data and validation data
```

### parameters:

```
doc_vector：path of original doc vector(can be tfidf vector or embedding vector)
doc_vec_train:path of training data of doc vectors
doc_vec_validation:path of validation data of doc vectors
```

## data_processor.To_kernel_vec()

### function：

```
get doc vector with strong features
```

### parameters:

```
doc_vec_train:path of training data of doc vectors
doc_vec_kernel：path of doc vectors with strong features
```

## navie_bayes_divider.__init__

### parameters:

```
doc_vec_train:path of training data of doc vectors
doc_vec_kernel：path of doc vectors with strong features
doc_vector：path of original doc vector(can be tfidf vector or embedding vector)
doc_vec_validation:path of validation data of doc vectors
navie_bayes_model_by_industry_dir_road:path of naive base model
predict_result_road：path of final prediction
```

### navie_bayes_divider.navie_bayes_model_train()

```
train all the naive bases divider for every two categories
```

### navie_bayes_divider.navie_bayes_predict()

```
predict the result of validation vectors
```

## svm_divider.__init__

### parameters:

```
doc_vec_train:path of training data of doc vectors
doc_vec_kernel：path of doc vectors with strong features
doc_vector：path of original doc vector(can be tfidf vector or embedding vector)
doc_vec_validation:path of validation data of doc vectors
svm_model_by_industry_dir_road:path of svm model
predict_result_road：path of final prediction
```

### svm_divider.svm_model_train()

```
train all the svm divider for every two categories
```

### svm_divider.svm_predict()

```
predict the result of validation vectors
```

