BEGIN{
    FS="\t"
    OFS="\t"
}

function compare_freq(i1,v1,i2,v2){
    if(v1!=v2){
        return (v1>v2)?-1:1
    }

    return (i1<i2)?-1:(i1>i2?1:0)
}


{
    for(i=1;i<=NF;i++){
        if($i!=""){
            freq[$i]++
        }
    }
}

END{
    n=asorti(freq,sorted_words,"compare_freq")

    for(i=1;i<=n;i++){
        word=sorted_words[i]
        print i-1,word,freq[word]
    }
}