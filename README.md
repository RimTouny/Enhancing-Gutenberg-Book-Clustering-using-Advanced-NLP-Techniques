# Enhancing-Gutenberg-Book-Clustering-using-Advanced-NLP-Techniques
Text clustering, an unsupervised ML technique in NLP, groups similar texts based on content. Techniques like hierarchical, k-means, or density-based clustering categorize unstructured data, unveiling insights and patterns in diverse datasets. This exploration was part of the NLP course in my University of Ottawa master's program in 2023.

  - Required libraries: scikit-learn, pandas, matplotlib.
  - Execute cells in a Jupyter Notebook environment.

## Unsupervised Text Clustering problem 
Text clustering involves grouping comparable texts based on content similarity, a crucial unsupervised technique.(chose 5 differnet books for 5 differnet author and genre)

```python
selected_books=['austen-emma.txt','whitman-leaves.txt','milton-paradise.txt', 'melville-moby_dick.txt','chesterton-thursday.txt']
```

## **Key Tasks Undertaken**

1. **Data Preparation, Preprocessing and, Cleaning:**
   - Listing all the books in Gutenberg’s library.
     ```python
     {'austen-emma.txt': 'Jane Austen',
     'austen-persuasion.txt': 'Jane Austen',
     'austen-sense.txt': 'Jane Austen',
     'carroll-alice.txt': 'Lewis Carroll',
     'chesterton-ball.txt': 'G.K. Chesterton',
     'chesterton-brown.txt': 'G. K. Chesterton',
     'chesterton-thursday.txt': 'G. K. Chesterton',
     'edgeworth-parents.txt': 'Maria Edgeworth',
     'melville-moby_dick.txt': 'Dick  Herman Melville',
     'shakespeare-caesar.txt': 'William Shakespeare',
     'shakespeare-hamlet.txt': 'William Shakespeare',
     'whitman-leaves.txt': 'Walt Whitman'}
     ```
   - Choose five different books by five different authors belong to the same category (History).
   - Data preparation:
      + Removing stop words.
      + Converting all words to the lower case.
      + Tokenize the text.
      +  Lemmatization is the next step that reduces a word to its base form.

   - Data Partitioning: partition each book into 200 documents, each document is a 100 word record.
     ![image](https://github.com/RimTouny/Enhancing-Gutenberg-Book-Clustering-using-Advanced-NLP-Techniques/assets/48333870/1ef28c48-fb0e-441f-940c-08b0f3edd2c3)


   - Data labeling as follows:
      +  austen-emma→ a
      + chesterton-thursday→ b
      +  shakespeare-hamlet→ c
      +  chesterton-ball→ d
      + carroll-alice→ e
    
    - Word Cloud Generation: Generates word clouds displaying the most frequent 100 words in books for each author.
      ![image](https://github.com/RimTouny/Enhancing-Gutenberg-Book-Clustering-using-Advanced-NLP-Techniques/assets/48333870/623c32b1-a2e1-446a-af29-3f552bd43d8e)

2. **Feature Engineering:**
   - Transformation
     + Bag of Word (BOW):It represents the occurrence of words within a document, it involves two things:
        * A vocabulary of known words.
        * A measure of the presence of known words.
     + Term Frequency - Inverse Document Frequency (TF-IDF):a technique to quantify words in a set of documents. We compute         a score for each word to signify its importance in the document and corpus.
     + Latent Dirichlet Allocation (LDA)
     + Word Embedding (Word2Vec)

![merge_from_ofoct](https://github.com/RimTouny/Enhancing-Gutenberg-Book-Clustering-using-Advanced-NLP-Techniques/assets/48333870/57f76f7b-13f6-4c18-8bfd-0e32a9207645)

   - Encoding
     
3. **Modeling:**: For each technique of the above, these following models are trained and tested.
   + K-Means
   + Expectation Maximization (EM)
   + Hierarchical clustering (Agglomerative) 

4. **Model Evaluation**
   - using Silhouette Score

     ![image](https://github.com/RimTouny/Enhancing-Gutenberg-Book-Clustering-using-Advanced-NLP-Techniques/assets/48333870/2e7b641f-6d1c-4e2d-9893-a965c68dab27)

     ![image](https://github.com/RimTouny/Enhancing-Gutenberg-Book-Clustering-using-Advanced-NLP-Techniques/assets/48333870/c6d455cb-7886-4561-baa6-69f708083c83)

   - using Kappa Scorce

     ![image](https://github.com/RimTouny/Enhancing-Gutenberg-Book-Clustering-using-Advanced-NLP-Techniques/assets/48333870/22e9c118-1f74-4e0e-916c-6d47fe8e721e)

     ![image](https://github.com/RimTouny/Enhancing-Gutenberg-Book-Clustering-using-Advanced-NLP-Techniques/assets/48333870/a4f3b004-2186-4863-b8f6-e0873037d883)

5. Champion Model
   - on Silhouette Score

     ![image](https://github.com/RimTouny/Enhancing-Gutenberg-Book-Clustering-using-Advanced-NLP-Techniques/assets/48333870/ff17d2e0-4db0-4ae1-8806-268210550f78)

   - on Kappa Scorce

     ![image](https://github.com/RimTouny/Enhancing-Gutenberg-Book-Clustering-using-Advanced-NLP-Techniques/assets/48333870/b549c004-6b77-4885-b8ab-18b72bac1698)

6. **Error Analysis of Champion Model**:
  - By reducing the number of clusters from 5 to 3
    + on Silhouette Score

       ![image](https://github.com/RimTouny/Enhancing-Gutenberg-Book-Clustering-using-Advanced-NLP-Techniques/assets/48333870/83bc1841-0683-4202-8669-a07996f78e01)

       ![image](https://github.com/RimTouny/Enhancing-Gutenberg-Book-Clustering-using-Advanced-NLP-Techniques/assets/48333870/01dd9d40-0124-4176-833f-7504ee160aac)

       
        - Champion Model

          ![image](https://github.com/RimTouny/Enhancing-Gutenberg-Book-Clustering-using-Advanced-NLP-Techniques/assets/48333870/0e213fd6-a895-4ed3-b079-775386b605df)

    + on Kappa Scorce

      ![image](https://github.com/RimTouny/Enhancing-Gutenberg-Book-Clustering-using-Advanced-NLP-Techniques/assets/48333870/57d7b3d8-717c-4cc7-93b6-54a584fc1dfc)
   
      ![image](https://github.com/RimTouny/Enhancing-Gutenberg-Book-Clustering-using-Advanced-NLP-Techniques/assets/48333870/46871156-3322-4591-9383-24aeb0b40134)


       - Champion Model
     
         ![image](https://github.com/RimTouny/Enhancing-Gutenberg-Book-Clustering-using-Advanced-NLP-Techniques/assets/48333870/3b0d6ee9-dbcc-4f0c-b66b-cf366505573e)




