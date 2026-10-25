BEGIN{
    FS="\t"
}

NF>=3{
    print $3
}