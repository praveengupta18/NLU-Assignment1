
# coding: utf-8

# In[1]:


import nltk
from nltk.corpus import gutenberg


# In[2]:


guten = gutenberg.sents()


# In[3]:


gt=list(guten)


# In[54]:


UNK = '<unk>'
START = '<s>'
STOP = '</s>'

def unigram(train):
    uni = []
    unks = set()
    for i in range(len(train)):
        sen = [w for w in train[i] if w not in [':','/',';','|',"''",'``','(',')','-','--','_','"',',','?']]
        sen.insert(0,START)  
        sen.insert(0,START)
        sen.append(STOP)
        uni += sen

    uni_cfd = nltk.FreqDist(uni)

    for word,freq in uni_cfd.items():
        if freq == 1:
            unks.add(word)

    for word in unks:
        del uni_cfd[word]

    uni_cfd[UNK] = len(unks)
    
    return unks,uni_cfd


# In[55]:


def bi_and_tri(train,unks):
    bi = []
    tri_ls = []
    for i in range(len(train)):
        sen = [w if w not in unks else UNK for w in train[i] if w not in ['?',':','/',';','|',"''",'``','(',')','-','--','_','"',',']]
        sen.insert(0,START)  
        sen.insert(0,START)
        sen.append(STOP)

        x = nltk.bigrams(sen)
        bi += list(x)

        x = nltk.trigrams(sen)
        tri_ls.append(list(x))

    tri = []

    for i in range(len(tri_ls)):
        l = [((a,b),c) for (a,b,c) in tri_ls[i]]
        tri += l

    bi_cfd = nltk.ConditionalFreqDist(bi)
    
    tri_cfd = nltk.ConditionalFreqDist(tri)
    
    return bi_cfd,tri_cfd


# In[56]:


gt_unks,gt_uni_cfd = unigram(gt)
gt_bi_cfd,gt_tri_cfd = bi_and_tri(gt,gt_unks)


# In[78]:


from random import randrange

def generate_model(unks, uni_cfd, bi_cfd, tri_cfd, word, lenght=10):
    count=0
    prnt=True
    unks=list(unks)
    while(count<lenght):
        if(word[1]!=START and word[1]!=STOP and prnt):
            print(word[1], end=' ')
            count+=1
        
        if(word[1]==STOP):
            word=(START,START)
            
        num = min(3,len(tri_cfd[word]))
        x = randrange(num)
        word1 = tri_cfd[word].most_common()[x][0]
        
        if(word1==UNK and num==1):
            num = min(3,len(bi_cfd[word[1]]))
            x = randrange(num)
            word1 = bi_cfd[word[1]].most_common()[x][0]
    
            if(word1==UNK and num==1):
                num=min(200,len(nui_cfd))
                x=randrange(30,num)
                word1=uni_cfd.most_common()[x]
            
        if(word1!=UNK):
            word = (word[1],word1)
            prnt=True
            
        if(word1==UNK):
            prnt=False


# In[90]:


generate_model(gt_unks,gt_uni_cfd,gt_bi_cfd,gt_tri_cfd,(START,START))

