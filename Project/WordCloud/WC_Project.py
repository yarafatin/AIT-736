
import pandas as pd
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt



data = pd.read_csv("train.csv")





# data.head()




# data.info()





# data.describe()





# data.shape





# len(data)




# data.values[0:4]




target_is_1 = data[data.target == 1]
target_is_0 = data[data.target == 0]




len(target_is_1)





len(target_is_0)





len(data)





# len(target_is_1) + len(target_is_0) == len(data)





#remove column qid and save as data2
data.columns
data2 = data.drop('qid', axis=1)




# % of insincere questions
qwe = round((len(target_is_1) / len(data) ),4)





print('Proportion of Questions Labeled as Insincere: ' + str(round((qwe*100),2))+"%")
print('Proportion of Questions Labeled as Sincere: ' + str(round(((1-qwe)*100),2))+"%")



from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



comment_words1 = ''
comment_words0 = ''
# stopwords = set(STOPWORDS)
# stopwords = ['nan', 'NaN', 'Nan', 'NAN'] + list(STOPWORDS)



# Create stopword list:
stopwords = set(STOPWORDS)
stopwords.update(["will", "thing", "way", "make", "take", "", "think", "one", "much", "u", "many", "want", "know", "going", "s", "", "", "", "", "", "", "", "", ""])




len(stopwords)



#trying to introduce a more robust list of stopwords
stopwords2 = pd.read_csv("stop_words_english.txt", header=None)





stopwords2 = set(stopwords2[0])



type(stopwords2)





len(stopwords)





len(stopwords2)





stopwords = stopwords.union(stopwords2)





len(stopwords)




#fake way to subset data - using this to reduce the # of rows to test the wordcloud
# dt1 = target_is_1[0:100000]

#full dataset
dt1 = target_is_1



values = dt1['question_text'].values

for val in values: 
    val = str(val) 
    tokens = val.split() 
    for i in range(len(tokens)): 
        tokens[i] = tokens[i].lower() 
    comment_words1 += ' '.join(tokens)+' '



text = """...""" # your text

text.split()
count = {}
for word in text.split():
    count.setdefault(word, 0)
    count[word] += 1

list_count = list(count.items())
list_count.sort(key=lambda i: i[1], reverse=True)
for i in list_count:
    print(i[0], ':', i[1])






color1 = 'white'

wordcloud1 = WordCloud(width=1000, height=600, 
            background_color=color1, 
            stopwords=stopwords,
            min_font_size=10).generate(comment_words1)




# Insincere Wordcloud
plt.figure(figsize=(15,12), facecolor=color1) 
plt.imshow(wordcloud1) 
plt.axis('off') 
plt.tight_layout(pad=2)




len(comment_words1)





#fake way to subset data - using this to reduce the # of rows to test the wordcloud
# dt0 = target_is_0[0:100000]

#full dataset
dt0 = target_is_0




values = dt0['question_text'].values

for val in values: 
    val = str(val) 
    tokens = val.split() 
    for i in range(len(tokens)): 
        tokens[i] = tokens[i].lower() 
    comment_words0 += ' '.join(tokens)+' '



color0 = 'white'

wordcloud0 = WordCloud(width=1000, height=600, 
            background_color=color0, 
            stopwords=stopwords,
            min_font_size=10).generate(comment_words0)



# Sincere Wordcloud
plt.figure(figsize=(15,12), facecolor=color0) 
plt.imshow(wordcloud0) 
plt.axis('off') 
plt.tight_layout(pad=2)




# import nltk
# from nltk.probability import FreqDist

# sentence="This is my sentence is is This is my sentence is is This is my sentence is is This is my sentence is is Once upon a time is is Once upon a time Both sincere and insincere questions have."

# tokens = nltk.tokenize.word_tokenize(comment_words1)
# fdist=FreqDist(tokens)



# type(fdist)





# fdist.plot(20,title='Title Name')