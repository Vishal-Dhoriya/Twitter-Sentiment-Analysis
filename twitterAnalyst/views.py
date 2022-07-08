import sys
import re
import string
import nltk
import warnings
import seaborn as sns
import tweepy
from textblob import TextBlob
from wordcloud import WordCloud
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import warnings
import string
from nltk.stem.porter import *
from nltk.stem.porter import *
from django.shortcuts import render

def index(request):
    return render(request,'index.html')

def analyse(request2):
    plt.style.use('fivethirtyeight')
    tweets_df = pd.read_csv('twitterAnalyst/static/train.csv')
    tweets_df
    tweets_df = tweets_df.drop(['id'],axis = 1)
    tweets_df
    consumerKey = "sMs6MjSkkhxXEYd4cHYd8wmfG"
    consumerSecret = "Sk3DsaLCxBaXuRaICjBsvTlvKrgHyLKRDM013phesxB7z8KsER"
    accessToken = "1543283400056139776-v3UN79J5PAwN5IhAPxgpsI8Y9cKIKb"
    accessTokenSecret = "kPkWCZMYjIzr120Pi5OEiBHagli9Uu7As7kuiHuew0pPq"
    auth = tweepy.OAuthHandler(consumer_key = consumerKey, consumer_secret=consumerSecret)
    auth.set_access_token(accessToken, accessTokenSecret)
    api = tweepy.API(auth)


    username = request2.POST.get('name', 'default')
    twtextract = int(request2.POST.get('name2', '10'))
    namesea = username
    # Extracts tweets from the twitter user 
    posts = api.user_timeline (screen_name = namesea, count= twtextract, lang="en", tweet_mode="extended")

    sto = twtextract
        
    #Create a dataframe with a column called Tweets 
    df = pd.DataFrame( [tweet.full_text for tweet in posts], columns=['tweet'])

    #Show the first 5 rows of data
    df.head(sto)

    #Create a function to get the subjectivity

    def getSubjectivity(text):
        return TextBlob (text).sentiment. subjectivity

    # Create a function to get the polarity

    def getPolarity (text):
        return TextBlob (text).sentiment.polarity

    #Create two new columns

    df[ 'Subjectivity'] = df[ 'tweet'].apply(getSubjectivity)
    df[ 'Polarity'] = df[ 'tweet'].apply(getPolarity)


    def getAnalysis(score):
        if score < 0:
            return 'Negative'
        elif score == 0:
            return 'Neutral'
        else:
            return 'Positive'

    df[ 'Analysis'] = df[ 'Polarity'].apply(getAnalysis)

    #Show the dataframe


    #Show the new dataframe with the new columns
    #df

    #Create a function to compute the negative, neutral and positive analysis
    def getAnalysiss(score):
        if score < 0:
            return 'Negative'
        elif score == 0:
            return 'Neutral'
        else:
            return 'Positive'

    df[ 'Analysis'] = df[ 'Polarity'].apply(getAnalysiss)

    #Show the dataframe
    def getAnalysis(score):
        if score < 0:
            return '1'
        elif score == 0:
            return '0'
        else:
            return '0'

    df[ 'label'] = df[ 'Polarity'].apply(getAnalysis)

    #Show the dataframe

    df

    # Plot the polarity and subjectivity
    plt.figure(figsize=(20,10))
    for i in range(0, df.shape[0]):
        plt.scatter(df['Polarity'][i], df[ 'Subjectivity'][i], color='Blue' )

    plt.title('Sentiment Analysis')
    plt.xlabel('Polarity')
    plt.ylabel('Subjectivity')
    plt.savefig('twitterAnalyst/static/polarity.png')
    # #plt.show()

    df = df.drop(['Polarity'], axis = 1)

    df = df.drop(['Subjectivity'], axis = 1)

    df = df.drop(['Analysis'], axis = 1)

    df

    #savng of file in csv formatd

    #from google.colab import files
    #uploaded = files.upload()
    df.to_csv('twitterAnalyst/static/test.csv',index =False)

    df['length'] = tweets_df['tweet'].apply(len)

    df

    df.describe()

    #df.hist(bins= 30, figsize =(13,5), color = 'r')

    #sns.countplot(df['label'], label = 'Count')

    #sns.countplot(df['length'], label = 'Count')

    #cleaning of tweets

    #Create a function to clean the tweets 
    def cleanTxt(text):
        text = re.sub (r'@[A-Za-z0-9]+', '', text) # Removed @mentions
        text = re.sub (r'#', '', text) #Removing the '#' symbol
        text = re.sub (r'RT[\s]+', '', text) # Removing RT 
        text = re.sub (r'https?:\/\/\S+', '', text) # Remove the hyper link
        return text

    def deEmojify(text):
        regrex_pattern = re.compile(pattern = "["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002500-\U00002BEF"  # chinese char
            u"\U00002702-\U000027B0"
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            u"\U0001f926-\U0001f937"
            u"\U00010000-\U0010ffff"
            u"\u2640-\u2642" 
            u"\u2600-\u2B55"
            u"\u200d"
            u"\u23cf"
            u"\u23e9"
            u"\u231a"
            u"\ufe0f"  # dingbats
            u"\u3030"
            
                            "]+", flags = re.UNICODE)
        return regrex_pattern.sub(r'',text)


    

    #Cleaning the text

    df['tweet']= df[ 'tweet'].apply(cleanTxt)
    df['tweet']= df[ 'tweet'].apply(deEmojify)
    #Show the cleaned text

    df


    # Plot The Word Cloud
    allWords = ' '.join( [twts for twts in df['tweet']] )
    # wordCloud = WordCloud(width = 500, height=300, random_state = 21, max_font_size = 119).generate(allWords)


    # plt.imshow(wordCloud, interpolation = "bilinear")
    # plt.axis('off')
    # #plt.show()



    train = pd.read_csv('twitterAnalyst/static/train.csv')
    test = pd.read_csv('twitterAnalyst/static/test.csv')

    comb=test.append(test,ignore_index=True) #train and test dataset are combined
    comb.shape

    def remove_pattern(input_text,pattern):
        r= re.findall(pattern, input_text)
        for i in r:
            input_text = re.sub(i, '', input_text)
        return input_text

    comb['tidy_tweet'] = np.vectorize(remove_pattern)(comb['tweet'],"@[\w]*") 
    comb.head()

    comb['tidy_tweet'] = comb['tidy_tweet'].str.replace("[^a-zA-Z#]"," ")
    comb.head(10)

    comb['tidy_tweet'] = comb['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3])) 

    tokenize_tweet = comb['tidy_tweet'].apply(lambda x:x.split()) #it will split all words by whitespace
    tokenize_tweet.head()

    stemmer = PorterStemmer()
    tokenize_tweet = tokenize_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) #it will stemmatized all words in tweet

    #now let's combine these tokens back

    for i in range(len(tokenize_tweet)):
        tokenize_tweet[i] = ' '.join(tokenize_tweet[i]) #concat all words into one sentence
    comb['tidy_tweet'] = tokenize_tweet

    all_word = ' '.join([text for text in comb['tidy_tweet']]) 
    from wordcloud import WordCloud
    wordcloud = WordCloud(width=800,height=500,random_state=21,max_font_size=110).generate(all_word)
    plt.figure(figsize=(10,7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    
    normal_word= ' '.join([text for text in comb['tidy_tweet'][comb['label']==0]])
    wordcloud= WordCloud(width=800,height=500,random_state=21,max_font_size=110).generate(normal_word)
    plt.figure(figsize=(10,7))
    plt.imshow(wordcloud,interpolation='bilinear')
    plt.axis('off')
    plt.savefig('twitterAnalyst/static/poswords.png')

    negative_word= ' '.join([text for text in comb['tidy_tweet'][comb['label']==1]])
    wordcloud= WordCloud(width=800,height=500,random_state=21,max_font_size=110).generate(negative_word)
    plt.figure(figsize=(10,7))
    plt.imshow(wordcloud,interpolation='bilinear')
    plt.axis('off')
    plt.savefig('twitterAnalyst/static/negwords.png')

    #collect hashtags

    # def hashtag_ext(x):
    #     hashtags=[]
    #     for i in x: #loop over words contain in tweet
    #         ht = re.findall(r"#(\w+)",i)
    #         hashtags.append(ht)
    #     return hashtags

    # #extracting hashtags from non racist tweets
    # ht_regular = hashtag_ext(comb['tidy_tweet'][comb['label']==0])
    # #extracting hashtags from racist tweets
    # ht_negative=hashtag_ext(comb['tidy_tweet'][comb['label']==1])
    # ht_regular = sum(ht_regular,[])
    # ht_negative = sum(ht_negative,[])



    # good_tweets = nltk.FreqDist(ht_regular)
    # df1 = pd.DataFrame({'Hashtag': list(good_tweets.keys()),'Count':list(good_tweets.values())})

    # #selecting top 20 most frequent hashtags
    # df1 = df1.nlargest(columns="Count",n=20)
    # plt.figure(figsize=(16,5))
    # ax = sns.barplot(data=df1, x="Hashtag", y="Count")
    # ax.set(ylabel = "Count")
    # plt.savefig('twitterAnalyst/static/postags.png')

    # bad_tweets = nltk.FreqDist(ht_negative)
    # df2 = pd.DataFrame({'Hashtag': list(bad_tweets.keys()),'Count': list(bad_tweets.values())}) #count number of occurrence of particular word

    # #selecting top 20 frequent  hashtags

    # df2 = df2.nlargest(columns = "Count",n=20)
    # plt.figure(figsize=(16,5))
    # ax1 = sns.barplot(data=df2)
    # ax1.set(ylabel = "Count")
    # plt.savefig('twitterAnalyst/static/negtags.png')

    





    combine=train.append(test,ignore_index=True) #train and test dataset are combined
    combine.shape

    #cleaning of tweets

    #Create a function to clean the tweets 
    def cleanTxt(text):
        text = re.sub (r'@[A-Za-z0-9]+', '', text) # Removed @mentions
        text = re.sub (r'#', '', text) #Removing the '#' symbol
        text = re.sub (r'RT[\s]+', '', text) # Removing RT 
        text = re.sub (r'https?:\/\/\S+', '', text) # Remove the hyper link
        return text

    def deEmojify(text):
        regrex_pattern = re.compile(pattern = "["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002500-\U00002BEF"  # chinese char
            u"\U00002702-\U000027B0"
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            u"\U0001f926-\U0001f937"
            u"\U00010000-\U0010ffff"
            u"\u2640-\u2642" 
            u"\u2600-\u2B55"
            u"\u200d"
            u"\u23cf"
            u"\u23e9"
            u"\u231a"
            u"\ufe0f"  # dingbats
            u"\u3030"
            
                            "]+", flags = re.UNICODE)
        return regrex_pattern.sub(r'',text)


    

    #Cleaning the text

    df['tweet']= df[ 'tweet'].apply(cleanTxt)
    df['tweet']= df[ 'tweet'].apply(deEmojify)
    #Show the cleaned text

    df


    def remove_pattern(input_text,pattern):
        r= re.findall(pattern, input_text)
        for i in r:
            input_text = re.sub(i, '', input_text)
        return input_text

    combine['tidy_tweet'] = np.vectorize(remove_pattern)(combine['tweet'],"@[\w]*") 
    combine.head()

    combine['tidy_tweet'] = combine['tidy_tweet'].str.replace("[^a-zA-Z#]"," ")
    combine.head(10)

    combine['tidy_tweet'] = combine['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3])) 

    tokenized_tweet = combine['tidy_tweet'].apply(lambda x:x.split()) #it will split all words by whitespace
    tokenized_tweet.head()

    stemmer = PorterStemmer()
    tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) #it will stemmatized all words in tweet

    #now let's combine these tokens back

    for i in range(len(tokenized_tweet)):
        tokenized_tweet[i] = ' '.join(tokenized_tweet[i]) #concat all words into one sentence
    combine['tidy_tweet'] = tokenized_tweet

    # all_words = ' '.join([text for text in combine['tidy_tweet']]) 
    # from wordcloud import WordCloud
    # wordcloud = WordCloud(width=800,height=500,random_state=21,max_font_size=110).generate(all_words)
    # plt.figure(figsize=(10,7))
    # plt.imshow(wordcloud, interpolation="bilinear")
    # plt.axis('off')
    # #plt.show()

    # normal_words= ' '.join([text for text in combine['tidy_tweet'][combine['label']==0]])
    # wordcloud= WordCloud(width=800,height=500,random_state=21,max_font_size=110).generate(normal_words)
    # plt.figure(figsize=(10,7))
    # plt.imshow(wordcloud,interpolation='bilinear')
    # plt.axis('off')
    # #plt.show()

    # negative_words= ' '.join([text for text in combine['tidy_tweet'][combine['label']==1]])
    # wordcloud= WordCloud(width=800,height=500,random_state=21,max_font_size=110).generate(negative_words)
    # plt.figure(figsize=(10,7))
    # plt.imshow(wordcloud,interpolation='bilinear')
    # plt.axis('off')
    # #plt.show()

    #collect hashtags

    # def hashtag_extract(x):
    #     hashtags=[]
    #     for i in x: #loop over words contain in tweet
    #         ht = re.findall(r"#(\w+)",i)
    #         hashtags.append(ht)
    #     return hashtags

    # #extracting hashtags from non racist tweets
    # ht_regular = hashtag_extract(combine['tidy_tweet'][combine['label']==0])
    # #extracting hashtags from racist tweets
    # ht_negative=hashtag_extract(combine['tidy_tweet'][combine['label']==1])
    # ht_regular = sum(ht_regular,[])
    # ht_negative = sum(ht_negative,[])

    # #non-racist tweets

    # nonracist_tweets = nltk.FreqDist(ht_regular)
    # df1 = pd.DataFrame({'Hashtag': list(nonracist_tweets.keys()),'Count':list(nonracist_tweets.values())})

    # #selecting top 20 most frequent hashtags
    # df1 = df1.nlargest(columns="Count",n=20)
    # plt.figure(figsize=(16,5))
    # ax = sns.barplot(data=df1, x="Hashtag", y="Count")
    # ax.set(ylabel = "Count")
    # #plt.show()

    # racist_tweets = nltk.FreqDist(ht_negative)
    # df2 = pd.DataFrame({'Hashtag': list(racist_tweets.keys()),'Count': list(racist_tweets.values())}) #count number of occurrence of particular word

    # #selecting top 20 frequent  hashtags

    # df2 = df2.nlargest(columns = "Count",n=20)
    # plt.figure(figsize=(16,5))
    # ax = sns.barplot(data=df2, x="Hashtag",y="Count")
    # plt.savefig("zzzz.png")
    # #plt.show()

    # pip install gensim

    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer 
    import gensim 

    bow_vectorizer = CountVectorizer(max_df=0.90 ,min_df=2 , max_features=1000,stop_words='english')
    bow = bow_vectorizer.fit_transform(combine['tidy_tweet']) # tokenize and build vocabulary
    bow.shape

    combine=combine.fillna(0) #replace all null values by 0
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(bow, combine['label'],
                                                        test_size=0.2, random_state=69)

    print("X_train_shape : ",X_train.shape)
    print("X_test_shape : ",X_test.shape)
    print("y_train_shape : ",y_train.shape)
    print("y_test_shape : ",y_test.shape)

    from sklearn.naive_bayes import MultinomialNB  # Naive Bayes Classifier

    model_naive = MultinomialNB().fit(X_train, y_train) 
    predicted_naive = model_naive.predict(X_test)

    from sklearn.metrics import confusion_matrix

    plt.figure(dpi=600)
    mat = confusion_matrix(y_test, predicted_naive)
    sns.heatmap(mat.T, annot=True, fmt= 'd' ,cbar=True)

    plt.title('Confusion Matrix for Naive Bayes')
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    plt.savefig("twitterAnalyst/static/confusion_matrix.png")
    #plt.show()

    from sklearn.metrics import accuracy_score

    score_naive = accuracy_score(predicted_naive, y_test)
    print("Accuracy with Naive-bayes: ",score_naive)
    return render(request2,'index2.html')