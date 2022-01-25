from flask import Flask, render_template, request
import sys


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')  #route to the main index page

@app.route('/proc')
def isdf():
    with open('alex.txt','r', encoding='utf-8') as f:
        return render_template('index.html', text=f.read()) #fetch in template/index.html

@app.route('/output')
def outpudt():

    with open('SummarizedData.txt', 'r', encoding='cp1252') as a, open('RawText', 'r', encoding='cp1252') as b:
        return render_template('result.html', summarized=a.read(), original=b.read())


@app.route("/Analysing", methods=['POST'])
def my_form_post():
    df = request.form['text']
    lines = request.form['lines']
    f = open("RawText", "w", encoding='utf-8')
    f.write(df)
    f.close()

    ## Main Python file here
    #
    # 
    # 
    # 
    # 
    # 
    # 
    #  

    import numpy as np
    import pandas as pd
    import nltk
    import networkx as nx
    from nltk.tokenize import sent_tokenize
    from nltk.corpus import stopwords
    from sklearn.metrics.pairwise import cosine_similarity
    nltk.download('punkt') # one time execution
    import re


    
    
    sentences = []
    sentences.append(sent_tokenize(df))
    sentences = [y for x in sentences for y in x] # flattern list(Seperate on the basis of fullstop (.))
    # print(sentences)


    # Extract word vectors
    word_embeddings = {}
    f = open('glove.6B.100d.txt', encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = coefs
    f.close()



    # remove punctuations, numbers and special characters
    clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")

    # make alphabets lowercase
    clean_sentences = [df.lower() for s in clean_sentences]



    nltk.download('stopwords')
    
    stop_words = stopwords.words('english')


    # function to remove stopwords
    def remove_stopwords(sen):
        sen_new = " ".join([i for i in sen if i not in stop_words])
        return sen_new


    # remove stopwords from the sentences
    clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]
    # print(clean_sentences)


    # # Extract word vectors
    # word_embeddings = {}
    # f = open('glove.6B.100d.txt', encoding='utf-8')
    # for line in f:
    #     values = line.split()
    #     word = values[0]
    #     coefs = np.asarray(values[1:], dtype='float32')
    #     word_embeddings[word] = coefs
    # f.close()


    sentence_vectors = []
    for i in clean_sentences:
        if (len(i) != 0):
            v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
        else:
            v = np.zeros((100,))
        sentence_vectors.append(v)



    # similarity matrix
    sim_mat = np.zeros([len(sentences), len(sentences)])

    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]


    
    

    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph)


    ranked_sentences = sorted(((scores[i],df) for i,df in enumerate(sentences)), reverse=True)
   
    sys.stdout = open("ranked_wise_data.txt", "w", encoding="utf8")
    for i in range(0,len(ranked_sentences)):
        print(ranked_sentences[i][1])
        print("\n")
    sys.stdout.close()


    outputsss = []
    count=0
    # Extract top 10 sentences as the summary
    # for i in range (0,3):
    #     outputsss.append(ranked_sentences[0][1][i])

    
    # listToStr = ' '.join(map(str, outputsss))

    # final = open("SummarizedData", "w")
    # final.write(listToStr)
    # final.close()

    sys.stdout = open("SummarizedData.txt", "w", encoding="utf8")

    for i in range (0,int(lines)):
        print(ranked_sentences[i][1])
    
    sys.stdout.close()



   #
   # 
   # 
   # 
   # 
   # 
   # 
   #  
   ##End of Main python file




    return render_template('loading.html')
    


if __name__ == "__main__":
    app.run(debug=True)