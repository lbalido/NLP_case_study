# NLP_case_study

### About The Data
Stored in an S3 bucket in AWS
The data is a collection of nearly every English language Wikipedia page
The test file contained a total of 45,625 words, of those 13,706 were unique words
Performed on a subset set of data, 50,000 articles


### Cleaning The Data
Lowercase
Punctuations
Stop words
Spaces
URLs

### EDA
- Most common words


### Dense Representation
- Topic Analysis
- IDF

Goal: Determine relationships between words in corpus (Wikipedia documents)
Procedure: Create sparse representation of words via vectorization, decompose sparse matrix (X) with SVD to find  V_T (X = U * sigma * V_T)
How to interpret V_T: V_T tells us how the words are related; higher absolute values in V_T correspond to more significant relationships between words



