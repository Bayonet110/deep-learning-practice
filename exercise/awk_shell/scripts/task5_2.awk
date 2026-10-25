BEGIN{
    FS="\t"
    OFS="\t"
    oov_count=0
}

FNR==NR{
    vocab[$2]=$1
    next
}

{
    for(i=1;i<=NF;i++){
        word=$i

        if(word in vocab){
            printf "%s",vocab[word]
        }else{
            printf "OOV Error: '%s' at line %d, column %d\n", \
               word, FNR, i > "/dev/stderr"

            printf "oov"
            oov_count++
        }

        if(i<NF){
            printf "\t"
        }
    }

    printf "\n"
}

END{
    if(oov_count>0){
        printf "Total OOV errors: %d\n", oov_count > "/dev/stderr"
        exit 1
    }
}