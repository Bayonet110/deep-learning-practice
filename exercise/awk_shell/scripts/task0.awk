BEGIN{
    RS="[[:space:]]+"
    count=0
}

NF>0{
    if(count%10!=0){
        printf "\t"
    }

    printf "%s",$0
    count++

    if(count%10==0){
        printf "\n"
    }
}

END{
    if(count%10!=0){
        printf "\n"
    }
}