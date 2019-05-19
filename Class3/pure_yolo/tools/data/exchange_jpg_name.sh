# var=file.txt
# tmp=${var##*/}
# echo ${tmp%.*}
dataDir=$1
num=451

# for file in `ls $dataDir`
# do
#     if [ "$file" == "000001.JPEG" ]; then
#         mv $file "test.jpg"
#     fi
# done

for file in `ls $dataDir`
do
    echo "num :"$num
    # echo "$file"
    # echo "${file%.*}" # 文件前缀名
    # PreName=""
    # Len=${#PreName}

    NumLen=${#num}
    # echo $NumLen
    if [ "$NumLen" == "1" ]; then 
        PreName="00000"$num
        # echo $PreName
    elif  [ "$NumLen" == "2" ]; then
        PreName="0000"$num
        # echo $PreName
    else 
        PreName="000"$num
        # echo $PreName
    fi    
    let num++
     
    echo "Ori jpg Name :"$file
    mv $dataDir/$file $dataDir/$PreName".jpg"
    echo 

done

