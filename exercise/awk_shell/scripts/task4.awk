BEGIN {
    FS="\t"
    total_words=0
    total_rows=0
}

NF>=3{
    n=split($3,words,/ +/)
    total_words+=n
    total_rows++
}

END{
    if(total_rows>0){
        printf "Total rows:%d\n",total_rows
        printf "Total words in column 3:%d\n",total_words
        printf "Average words:%.6f\n",total_words/total_rows
    }else{
        printf "No Valid data."
    }
}